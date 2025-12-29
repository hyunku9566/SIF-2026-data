# IGZO TFT 시뮬레이션 데이터 패키지

이 폴더는 IGZO TFT (Indium Gallium Zinc Oxide Thin-Film Transistor)의 전기적 특성(Transfer Curve)을 시뮬레이션하여 생성한 데이터셋과 생성 스크립트를 포함하고 있습니다.

## 파일 구성

### 1. 데이터 생성 스크립트
- **`generate_igzo_devsim.py`**: 
  - IGZO TFT의 물리적 모델(Analytical Model)을 기반으로 Transfer Curve를 생성하는 Python 스크립트입니다.
  - DEVSIM 라이브러리가 설치되어 있는 경우 TCAD 수준의 정밀 시뮬레이션도 가능하도록 설계되었습니다. (현재는 Analytical Model 모드로 동작)
  - 실행 시 `generated_data` 폴더에 5,000개의 샘플 데이터를 생성합니다.

### 2. 생성된 데이터 (`generated_data/`)
스크립트 실행 결과로 생성된 데이터 파일들입니다.

- **`igzo_data_[날짜]_[시간].json`**: 
  - 생성된 5,000개 소자의 전체 데이터입니다.
  - 각 소자의 파라미터(W, L, Tox 등), 시뮬레이션 조건(Vds, Vgs 범위), 결과 전류(Ids), 추출된 특성(Vth, SS, On/Off Ratio 등)을 모두 포함합니다.
  
- **`igzo_data_[날짜]_[시간]_features.csv`**: 
  - 전체 데이터에서 주요 파라미터와 특성값(Feature)만 추출하여 정리한 CSV 파일입니다.
  - 데이터 분석 및 엑셀 확인 용도로 적합합니다.

- **`sample_curves.png`**: 
  - 생성된 데이터 중 일부 샘플의 Transfer Curve(Linear/Log scale)를 시각화한 이미지입니다.

- **`curves.npy`**, **`targets.npy`**: 
  - 딥러닝 모델 학습을 위해 전처리된 Numpy 배열 파일입니다.
  - `curves.npy`: (N, 2, 200) 형태의 입력 데이터 (Vgs, Ids 곡선)
  - `targets.npy`: (N, 5) 형태의 타겟 데이터 (Vth, Log(On/Off), SS, Mobility, Log(Ion))

## 사용 방법

### 환경 설정
필요한 Python 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

### 데이터 재생성 및 수량 변경
데이터를 다시 생성하거나 생성할 데이터의 개수를 변경하려면 `--samples` 옵션을 사용하세요.
또한 `--mode` 옵션을 통해 시뮬레이션 모드(tcad 또는 analytical)를 선택할 수 있습니다. (기본값: tcad)

**기본 실행 (TCAD 모드, 5,000개 생성):**
```bash
python generate_igzo_devsim.py
```

**Analytical 모드로 실행:**
```bash
python generate_igzo_devsim.py --mode=analytical
```

**50,000개 생성 예시:**
```bash
python generate_igzo_devsim.py --samples=50000
```

**100개 테스트 생성 예시:**
```bash
python generate_igzo_devsim.py --samples=100
```

## 시뮬레이션 파라미터 범위
생성된 데이터는 다음과 같은 범위의 랜덤 파라미터를 가집니다.
- Channel Width (W): 5 ~ 50 μm
- Channel Length (L): 2 ~ 20 μm
- Channel Thickness: 20 ~ 80 nm
- Gate Oxide Thickness: 50 ~ 200 nm
- Mobility: 5 ~ 40 cm²/Vs
- Threshold Voltage (Vth): -1.0 ~ 3.0 V
# SIF-2026-data
