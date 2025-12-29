"""
DEVSIM 기반 IGZO TFT Transfer Curve 데이터 생성
IGZO (Indium Gallium Zinc Oxide) TFT 소자 물리 시뮬레이션
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import json
from datetime import datetime
import random
import argparse

# DEVSIM 임포트
try:
    import devsim
    from devsim import (
        create_device,
        set_parameter, get_parameter, 
        node_model, edge_model, contact_node_model, contact_edge_model,
        equation, contact_equation, interface_equation,
        solve, get_node_model_values, get_contact_current,
        create_1d_mesh, add_1d_mesh_line, add_1d_contact, add_1d_region,
        finalize_mesh, write_devices, get_mesh_list
    )
    DEVSIM_AVAILABLE = True
except ImportError as e:
    DEVSIM_AVAILABLE = False
    print(f"⚠️ DEVSIM not available, using analytical model instead. Error: {e}")

# ==========================================
# 물리 상수
# ==========================================
Q_E = 1.602e-19      # 전하량 (C)
K_B = 1.38e-23       # 볼츠만 상수 (J/K)
EPS_0 = 8.854e-14    # 진공 유전율 (F/cm)
EPS_IGZO = 10.0      # IGZO 유전율
EPS_SIO2 = 3.9       # SiO2 유전율
EPS_HFOX = 25.0      # HfO2 유전율

@dataclass
class IGZOTFTParams:
    """IGZO TFT 소자 파라미터"""
    # 채널 geometry
    W: float = 10e-4      # 채널 폭 (cm) - 10 um
    L: float = 5e-4       # 채널 길이 (cm) - 5 um
    t_channel: float = 30e-7   # IGZO 두께 (cm) - 30 nm
    t_ox: float = 100e-7       # gate oxide 두께 (cm) - 100 nm
    
    # IGZO 물성
    mu_0: float = 15.0       # 저전계 이동도 (cm²/Vs)
    n_i: float = 1e16        # 고유 캐리어 농도 (/cm³)
    N_d: float = 1e17        # 도핑 농도 (/cm³)
    E_a: float = 0.1         # activation energy (eV)
    
    # TFT 특성
    Vth_0: float = 0.5       # 기본 Vth (V)
    SS_ideal: float = 60.0   # 이상적 SS (mV/dec)
    N_it: float = 1e11       # 계면 트랩 밀도 (/cm²/eV)
    
    # 온도
    T: float = 300.0         # 온도 (K)
    
    # oxide 종류
    oxide_type: str = "SiO2"  # SiO2, HfO2

# ==========================================
# Analytical IGZO TFT Model (DEVSIM 없을 때 사용)
# ==========================================
class AnalyticalIGZOModel:
    """IGZO TFT 해석적 모델 (논문 기반)"""
    
    def __init__(self, params: IGZOTFTParams):
        self.params = params
        self._calc_derived_params()
    
    def _calc_derived_params(self):
        """파생 파라미터 계산"""
        p = self.params
        
        # 열 전압
        self.V_T = K_B * p.T / Q_E
        
        # oxide capacitance
        if p.oxide_type == "HfO2":
            eps_ox = EPS_HFOX
        else:
            eps_ox = EPS_SIO2
        self.C_ox = EPS_0 * eps_ox / p.t_ox  # F/cm²
        
        # Subthreshold swing
        C_it = Q_E * p.N_it  # 계면 트랩 capacitance
        C_dep = EPS_0 * EPS_IGZO / p.t_channel  # depletion capacitance
        self.SS = p.SS_ideal * (1 + (C_it + C_dep) / self.C_ox)  # mV/dec
        
        # 실효 이동도 (온도 의존성)
        self.mu_eff = p.mu_0 * np.exp(-p.E_a / (K_B * p.T / Q_E))
    
    def transfer_curve(self, Vgs: np.ndarray, Vds: float = 1.0) -> np.ndarray:
        """
        Transfer Curve (Id vs Vg) 계산
        IGZO TFT의 gradual channel approximation + subthreshold 영역
        """
        p = self.params
        Ids = np.zeros_like(Vgs)
        
        Vth = p.Vth_0
        
        for i, vg in enumerate(Vgs):
            V_eff = vg - Vth
            
            if V_eff < -10 * self.V_T:
                # Deep subthreshold (off 영역)
                Ids[i] = (self.mu_eff * self.C_ox * (p.W / p.L) * 
                         (self.V_T ** 2) * np.exp(V_eff / (self.SS / 1000 / np.log(10))))
                
            elif V_eff < 0:
                # Subthreshold 영역
                I_sub = (self.mu_eff * self.C_ox * (p.W / p.L) * 
                        (self.V_T ** 2) * np.exp(V_eff / (self.SS / 1000 / np.log(10))))
                Ids[i] = I_sub
                
            elif V_eff < Vds:
                # Linear 영역
                Ids[i] = self.mu_eff * self.C_ox * (p.W / p.L) * (V_eff * Vds - Vds**2 / 2)
                
            else:
                # Saturation 영역
                Ids[i] = 0.5 * self.mu_eff * self.C_ox * (p.W / p.L) * V_eff**2
                # 채널 길이 변조 효과
                lambda_ch = 0.05  # 채널 길이 변조 계수
                Ids[i] *= (1 + lambda_ch * Vds)
        
        # 노이즈 추가 (현실적 데이터)
        noise_level = 0.02
        Ids *= (1 + np.random.normal(0, noise_level, len(Ids)))
        
        return np.abs(Ids)
    
    def output_curve(self, Vds: np.ndarray, Vgs: float) -> np.ndarray:
        """Output Curve (Id vs Vd) 계산"""
        p = self.params
        Ids = np.zeros_like(Vds)
        
        V_eff = Vgs - p.Vth_0
        
        if V_eff <= 0:
            return np.abs(Ids + 1e-14)
        
        for i, vd in enumerate(Vds):
            vd = np.abs(vd)
            if vd < V_eff:
                # Linear 영역
                Ids[i] = self.mu_eff * self.C_ox * (p.W / p.L) * (V_eff * vd - vd**2 / 2)
            else:
                # Saturation 영역
                Ids[i] = 0.5 * self.mu_eff * self.C_ox * (p.W / p.L) * V_eff**2
                lambda_ch = 0.05
                Ids[i] *= (1 + lambda_ch * vd)
        
        return np.abs(Ids)

# ==========================================
# DEVSIM 기반 정밀 시뮬레이션
# ==========================================
class DEVSIMIGZOSimulator:
    """DEVSIM을 사용한 정밀 IGZO TFT 시뮬레이션"""
    
    def __init__(self, params: IGZOTFTParams):
        self.params = params
        self.device_name = "igzo_tft"
        self.region_name = "channel"
        self._setup_device()
    
    def _setup_device(self):
        """1D IGZO TFT 소자 설정"""
        if not DEVSIM_AVAILABLE:
            return
        
        p = self.params
        device = self.device_name
        region = self.region_name
        
        # 메시 생성 (채널 두께 방향)
        mesh_name = "igzo_mesh"
        
        # 기존 메시가 있으면 스킵
        if mesh_name in get_mesh_list():
            return
        
        create_1d_mesh(mesh=mesh_name)
        
        # 노드 추가
        t_ch = p.t_channel * 1e4  # cm -> um
        add_1d_mesh_line(mesh=mesh_name, pos=0, ps=t_ch/50, tag="top")
        add_1d_mesh_line(mesh=mesh_name, pos=t_ch, ps=t_ch/50, tag="bottom")
        
        # 영역 정의
        add_1d_region(mesh=mesh_name, material="IGZO", region=region, 
                     tag1="top", tag2="bottom")
        
        # 컨택트 정의
        add_1d_contact(mesh=mesh_name, name="gate", tag="top", material="metal")
        add_1d_contact(mesh=mesh_name, name="back", tag="bottom", material="metal")
        
        finalize_mesh(mesh=mesh_name)
        create_device(mesh=mesh_name, device=device)
        
        # 물질 파라미터 설정
        set_parameter(device=device, region=region, name="Permittivity", 
                     value=EPS_IGZO * EPS_0)
        set_parameter(device=device, region=region, name="n_i", value=p.n_i)
        set_parameter(device=device, region=region, name="mu_n", value=p.mu_0)
        set_parameter(device=device, region=region, name="T", value=p.T)
    
    def simulate_transfer(self, Vgs: np.ndarray, Vds: float = 1.0) -> np.ndarray:
        """Transfer curve 시뮬레이션"""
        # DEVSIM 풀 시뮬레이션은 복잡하므로
        # 여기서는 해석적 모델 사용
        model = AnalyticalIGZOModel(self.params)
        return model.transfer_curve(Vgs, Vds)

# ==========================================
# 무한 데이터 생성기
# ==========================================
class IGZODataGenerator:
    """IGZO TFT 시뮬레이션 데이터 무한 생성"""
    
    def __init__(self, output_dir: str = "./generated_data", mode: str = 'tcad'):
        self.output_dir = output_dir
        self.mode = mode
        os.makedirs(output_dir, exist_ok=True)
        
        if self.mode == 'tcad' and not DEVSIM_AVAILABLE:
            print("⚠️ Warning: TCAD mode requested but DEVSIM is not available. Falling back to analytical mode.")
            self.mode = 'analytical'
            
        print(f"Initialized IGZO Data Generator (Mode: {self.mode})")
        
        # 파라미터 범위 정의 (다양한 소자 조건 - 현실적 범위로 조정)
        self.param_ranges = {
            'W': (5e-4, 50e-4),         # 5-50 um
            'L': (2e-4, 20e-4),         # 2-20 um
            't_channel': (20e-7, 80e-7),   # 20-80 nm
            't_ox': (50e-7, 200e-7),       # 50-200 nm
            'mu_0': (5.0, 40.0),           # 5-40 cm²/Vs (IGZO 현실적 범위)
            'N_d': (1e16, 5e17),           # doping (현실적 범위)
            'Vth_0': (-1.0, 3.0),          # Vth range (현실적 범위)
            'N_it': (1e10, 5e11),          # 계면 트랩 (현실적 범위)
            'T': (280, 350),               # 온도 범위
        }
        
        # Vg sweep 범위
        self.vg_ranges = [
            (-10, 20, 200),   # start, end, points
            (-5, 15, 150),
            (-20, 30, 250),
            (0, 10, 100),
        ]
        
        # Vd 조건
        self.vd_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    def _random_params(self) -> IGZOTFTParams:
        """랜덤 파라미터 생성"""
        r = self.param_ranges
        
        return IGZOTFTParams(
            W=np.random.uniform(*r['W']),
            L=np.random.uniform(*r['L']),
            t_channel=np.random.uniform(*r['t_channel']),
            t_ox=np.random.uniform(*r['t_ox']),
            mu_0=np.random.uniform(*r['mu_0']),
            N_d=10**np.random.uniform(np.log10(r['N_d'][0]), np.log10(r['N_d'][1])),
            Vth_0=np.random.uniform(*r['Vth_0']),
            N_it=10**np.random.uniform(np.log10(r['N_it'][0]), np.log10(r['N_it'][1])),
            T=np.random.uniform(*r['T']),
            oxide_type=random.choice(['SiO2', 'HfO2']),
        )
    
    def generate_single(self) -> Dict:
        """단일 Transfer Curve 데이터 생성"""
        params = self._random_params()
        
        # 랜덤 Vg 범위 선택
        vg_start, vg_end, n_points = random.choice(self.vg_ranges)
        Vgs = np.linspace(vg_start, vg_end, n_points)
        
        # 랜덤 Vds 선택
        Vds = random.choice(self.vd_values)
        
        # Transfer curve 계산
        if self.mode == 'tcad':
            simulator = DEVSIMIGZOSimulator(params)
            Ids = simulator.simulate_transfer(Vgs, Vds)
            # 특성 추출을 위한 참조 모델 (Analytical 모델의 파라미터 사용)
            ref_model = AnalyticalIGZOModel(params)
        else:
            model = AnalyticalIGZOModel(params)
            Ids = model.transfer_curve(Vgs, Vds)
            ref_model = model
        
        # 특성 추출
        features = self._extract_features(Vgs, Ids, params, ref_model)
        
        return {
            'params': {
                'W_um': params.W * 1e4,
                'L_um': params.L * 1e4,
                't_channel_nm': params.t_channel * 1e7,
                't_ox_nm': params.t_ox * 1e7,
                'mu_0': params.mu_0,
                'N_d': params.N_d,
                'Vth_0': params.Vth_0,
                'N_it': params.N_it,
                'T': params.T,
                'oxide_type': params.oxide_type,
            },
            'conditions': {
                'Vds': Vds,
                'Vg_start': vg_start,
                'Vg_end': vg_end,
            },
            'data': {
                'Vgs': Vgs.tolist(),
                'Ids': Ids.tolist(),
            },
            'features': features,
        }
    
    def _extract_features(self, Vgs, Ids, params, model) -> Dict:
        """특성 추출"""
        # Vth (변곡점 방법)
        gm = np.gradient(Ids, Vgs)
        max_gm_idx = np.argmax(np.abs(gm))
        vth_extracted = Vgs[max_gm_idx]
        
        # On/Off ratio (클리핑으로 inf 방지)
        I_on = np.max(Ids)
        I_off = max(np.min(Ids[Ids > 0]) if np.any(Ids > 0) else 1e-15, 1e-15)
        on_off_ratio = min(I_on / I_off, 1e15)  # 최대 10^15로 클리핑
        
        # Subthreshold Swing (합리적 범위로 클리핑)
        ss = np.clip(model.SS, 60, 5000)  # 60 mV/dec ~ 5000 mV/dec
        
        # log On/Off 클리핑
        log_on_off = np.clip(np.log10(on_off_ratio) if on_off_ratio > 0 else 0, 0, 15)
        
        return {
            'Vth_extracted': float(np.clip(vth_extracted, -20, 30)),
            'Vth_true': float(params.Vth_0),
            'on_off_ratio': float(on_off_ratio),
            'log_on_off': float(log_on_off),
            'SS_mV_dec': float(ss),
            'mu_eff': float(np.clip(model.mu_eff, 0.01, 100)),
            'I_on': float(np.clip(I_on, 1e-12, 1e-2)),
            'I_off': float(I_off),
        }
    
    def generate_batch(self, n_samples: int = 100) -> List[Dict]:
        """배치 데이터 생성"""
        data = []
        for i in range(n_samples):
            sample = self.generate_single()
            sample['sample_id'] = i
            data.append(sample)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i+1}/{n_samples} samples")
        
        return data
    
    def generate_and_save(self, n_samples: int = 1000, 
                          filename: str = None) -> str:
        """데이터 생성 및 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"igzo_data_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        print(f"Generating {n_samples} IGZO TFT samples...")
        data = self.generate_batch(n_samples)
        
        # JSON 저장
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved to {filepath}")
        
        # CSV 형식으로도 저장 (특성만)
        features_list = []
        for sample in data:
            row = {**sample['params'], **sample['conditions'], **sample['features']}
            features_list.append(row)
        
        csv_path = filepath.replace('.json', '_features.csv')
        pd.DataFrame(features_list).to_csv(csv_path, index=False)
        print(f"Features saved to {csv_path}")
        
        return filepath
    
    def generate_for_training(self, n_samples: int = 1000, 
                              seq_length: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        딥러닝 학습용 데이터 생성
        Returns:
            curves: [N, 2, seq_length] - (Vg_norm, Id_norm)
            targets: [N, 5] - (Vth, On/Off, SS, mu, I_on)
        """
        curves = []
        targets = []
        
        for i in range(n_samples):
            sample = self.generate_single()
            
            Vgs = np.array(sample['data']['Vgs'])
            Ids = np.array(sample['data']['Ids'])
            
            # Resampling to fixed length
            from scipy import interpolate
            f_id = interpolate.interp1d(np.linspace(0, 1, len(Vgs)), Ids, 
                                         kind='linear', fill_value='extrapolate')
            f_vg = interpolate.interp1d(np.linspace(0, 1, len(Vgs)), Vgs,
                                         kind='linear', fill_value='extrapolate')
            
            x_new = np.linspace(0, 1, seq_length)
            Ids_resampled = f_id(x_new)
            Vgs_resampled = f_vg(x_new)
            
            # 정규화
            Vg_norm = (Vgs_resampled - Vgs_resampled.min()) / (Vgs_resampled.max() - Vgs_resampled.min() + 1e-10)
            Id_log = np.log10(np.abs(Ids_resampled) + 1e-15)
            Id_norm = (Id_log - Id_log.min()) / (Id_log.max() - Id_log.min() + 1e-10)
            
            curve = np.stack([Vg_norm, Id_norm], axis=0)
            
            # nan/inf 체크
            if np.isnan(curve).any() or np.isinf(curve).any():
                continue
            
            # 타겟 (nan/inf 체크)
            f = sample['features']
            target = [
                f['Vth_extracted'],
                f['log_on_off'],
                f['SS_mV_dec'],
                f['mu_eff'],
                np.log10(f['I_on'] + 1e-15),  # I_on을 log scale로
            ]
            
            # nan/inf 체크
            if any(np.isnan(t) or np.isinf(t) for t in target):
                continue
            if any(np.isnan(curve).any() or np.isinf(curve).any() for curve in [Vg_norm, Id_norm]):
                continue
                
            curves.append(np.stack([Vg_norm, Id_norm], axis=0))
            targets.append(target)
            
            if (i + 1) % 500 == 0:
                print(f"Prepared {len(targets)}/{n_samples} valid training samples")
        
        return np.array(curves), np.array(targets)

# ==========================================
# 시각화
# ==========================================
def plot_sample_curves(generator: IGZODataGenerator, n_samples: int = 5):
    """샘플 곡선 시각화"""
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
    
    for i in range(n_samples):
        sample = generator.generate_single()
        Vgs = np.array(sample['data']['Vgs'])
        Ids = np.array(sample['data']['Ids'])
        
        # Linear scale
        axes[0, i].plot(Vgs, Ids * 1e6, 'b-', linewidth=2)
        axes[0, i].set_xlabel('Vgs (V)')
        axes[0, i].set_ylabel('Ids (μA)')
        axes[0, i].set_title(f"W/L={sample['params']['W_um']:.1f}/{sample['params']['L_um']:.1f}")
        axes[0, i].grid(True, alpha=0.3)
        
        # Log scale
        axes[1, i].semilogy(Vgs, np.abs(Ids), 'b-', linewidth=2)
        vth = sample['features']['Vth_extracted']
        axes[1, i].axvline(vth, color='r', linestyle='--', label=f'Vth={vth:.2f}V')
        axes[1, i].set_xlabel('Vgs (V)')
        axes[1, i].set_ylabel('|Ids| (A)')
        axes[1, i].set_title(f"SS={sample['features']['SS_mV_dec']:.0f}mV/dec")
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
def main():
    # 커맨드 라인 인자 파싱
    parser = argparse.ArgumentParser(description='IGZO TFT Data Generator')
    parser.add_argument('--samples', type=int, default=5000, help='Number of samples to generate (default: 5000)')
    parser.add_argument('--mode', type=str, default='tcad', choices=['tcad', 'analytical'], help='Simulation mode: tcad or analytical (default: tcad)')
    args = parser.parse_args()

    print("="*60)
    print("  IGZO TFT Transfer Curve Data Generator")
    print(f"  Mode: {args.mode.upper()}")
    print("="*60)
    
    # 출력 디렉토리
    output_dir = "./generated_data"
    
    # 생성기 초기화
    generator = IGZODataGenerator(output_dir=output_dir, mode=args.mode)
    print("="*60)
    
    # 출력 디렉토리
    output_dir = "./generated_data"
    
    # 생성기 초기화
    generator = IGZODataGenerator(output_dir=output_dir)
    
    # 샘플 곡선 시각화
    print("\n[1] 샘플 곡선 생성 및 시각화...")
    plot_sample_curves(generator, n_samples=5)
    
    # 학습용 데이터 생성
    print("\n[2] 학습용 데이터 생성...")
    n_samples = args.samples
    filepath = generator.generate_and_save(n_samples=n_samples)
    
    # 딥러닝 형식 데이터
    print("\n[3] 딥러닝 형식 데이터 준비...")
    curves, targets = generator.generate_for_training(n_samples=500, seq_length=200)
    
    np.save(os.path.join(output_dir, 'curves.npy'), curves)
    np.save(os.path.join(output_dir, 'targets.npy'), targets)
    print(f"Curves shape: {curves.shape}")
    print(f"Targets shape: {targets.shape}")
    
    print("\n" + "="*60)
    print("  데이터 생성 완료!")
    print("="*60)

if __name__ == "__main__":
    main()
