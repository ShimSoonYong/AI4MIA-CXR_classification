# AI4MIA Week 5: COVID-19 Image Classification Project

**Author:** Shim Soonyong
**Date:** 2026/10/01

## Overview

본 프로젝트는 주어진 Chest X-ray 이미지를 활용하여 COVID-19 환자의 상태를 분류하는 **Convolutional Neural Network (CNN)** 모델을 개발하고 평가하는 것을 목표로 합니다.

이 프로젝트는 다음의 세 가지 주요 실험을 통해 진행됩니다:
1.  **Scratch CNN 구현**: 기본적인 CNN 구조를 직접 설계하고 학습합니다.
2.  **Transfer Learning**: 사전 학습된 강력한 모델(ResNet, DenseNet 등)을 활용하여 성능을 극대화합니다.
3.  **평가 분석**: Sensitivity, Specificity, AUROC 등 의료 영상 분류에 필수적인 지표를 계산하여 모델의 임상적 유효성을 검증합니다.

## Project Structure

본 프로젝트는 재현성과 모듈화를 위해 다음과 같은 구조로 구성되어 있습니다.

```
assignment1/
├── dataset/                   # 데이터셋 폴더 (.gitignore로 제외됨)
│   ├── raw/                # 원본 이미지 파일 (.png) 저장소
│   │   ├── covid/
│   │   ├── normal/
│   │   └── viral_pneumonia/
│   └── processed/          # 전처리된 데이터셋 (학습에 사용될 Tensor 형태 또는 이미지)
├── notebooks/              # 실험 및 시각화 결과 (Jupyter Notebooks)
│   ├── EDA.ipynb        # 데이터 분포 확인 및 이미지 시각화 분석
├── src/                    # 재사용 가능한 소스 코드 파일 (.py)
│   ├── __init__.py
│   ├── main.py             # 데이터 로더, 아규먼트 처리, 전체 파이프라인 실행 로직
│   ├── dataset.py          # Custom Dataset, DataLoader 정의 (데이터 로딩 및 증강 담당)
│   ├── models.py           # 모델 구조 정의 (Scratch CNN 및 Transfer Learning 모델 클래스 포함)
│   ├── train.py            # 학습 루프와 Trainer 클래스 정의
│   └── utils.py            # 지표 계산 함수 (Sensitivity, Specificity, AUROC 등)
├── results/                # 실험 결과 저장소
│   ├── checkpoints/        # 학습된 모델 가중치 파일 (.pth) 저장
│   ├── figures/            # ROC Curve, Confusion Matrix 등 시각화 이미지 저장
│   └── metrics.json        # 각 모델의 최종 성능 지표 기록 (AUROC, F1-score 등)
├── requirements.txt        # 프로젝트 실행에 필요한 모든 Python 패키지 목록
├── .gitignore              # Git 추적에서 제외할 파일 설정
└── README.md               # 본 문서
```

## Getting Started

### 1. 환경 준비 (Setup Environment)
프로젝트를 실행하기 전에 다음 단계를 수행해야 합니다.

1.  **코드 복사**: `src/` 폴더의 모든 `.py` 파일을 프로젝트 디렉토리 내에 생성합니다.
2.  **데이터셋 준비**: `data/raw/` 경로에 필요한 모든 Chest X-ray 이미지 파일(COVID, Normal, Viral Pneumonia)을 정리하여 배치합니다.
3.  **환경 설정**: 다음 명령어로 필요한 라이브러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

### 2. 데이터 준비 (Data Prearation)
`notebooks/01_EDA.ipynb`를 사용하여 원본 데이터를 확인하고, `src/dataset.py`에서 정의한 로직에 맞게 데이터를 전처리하여 `data/processed/` 폴더에 저장합니다.

### 3. 모델 학습 및 평가 (Training & Evaluation)
각 실험은 `notebooks/02_scratch_cnn.ipynb`와 `notebooks/03_transfer_learning.ipynb`에서 수행됩니다.

*   **학습**: `src/train.py`를 통해 정의된 Trainer 클래스를 사용하여 모델을 학습시킵니다.
*   **평가**: 학습 완료 후, `src/utils.py`의 함수들을 호출하여 **Sensitivity, Specificity, AUROC** 등의 지표를 계산하고 결과를 `results/metrics.json`에 기록합니다.
