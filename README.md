# 📊 Statics & Materials Analysis Pipeline

> **LabVIEW 측정 데이터 기반 응력-변형률(Stress-Strain) 자동 분석 및 시각화 파이프라인**  
> 기계공학 도메인 지식(재료/정역학)과 Python 데이터 분석 기술을 결합하여, 실험 원본 데이터 처리부터 물성치 추정, 리포트 생성까지 일괄 자동화한 프로젝트입니다.

---

## 🎯 Project Background & Objectives (개발 배경)

- **문제점 (Pain Point)**: LabVIEW 장비에서 산출되는 `.lvm` 파일은 비표준 텍스트 포맷 구조를 가지고 있어, 실험 후 엑셀을 이용한 수작업 가공 시 **많은 시간 소요 및 인간적 오차(Human Error)**가 발생함.
- **해결 방안 (Solution)**: Python 기반의 자동화 파이프라인을 구축하여 **비표준 데이터 파싱, 응력-변형률 계산, 머신러닝 기반 탄성 계수 추정, 시각화 및 CSV 리포트 저장**을 단 한 번의 스크립트 실행으로 처리하도록 구현.

---

## 🛠️ Tech Stack & Key Libraries

- **Language**: `Python 3.x`
- **Data Engineering**: `Pandas`, `NumPy` (비표준 `.lvm` 파싱, 수치 산출 및 배열 연산)
- **Machine Learning**: `Scikit-learn` (선형 회귀 기반 탄성 계수/Young's Modulus 자동 추정)
- **Visualization**: `Matplotlib` (응력-변형률 선도 Graph 자동 매핑 및 PNG 내보내기)

---

## 💡 Key Engineering Features (핵심 구현 기능)

### 1. 비표준 데이터 전처리 및 파싱 (Data Parsing)
- Complex Header 및 비정형 텍스트 구조를 가진 `.lvm` 원본 데이터를 분석하여 정형화된 Dataframe 구조로 자동 추출.

### 2. 정역학 / 재료역학 공식 기반 물성치 변환 (Physical Property Calculation)
- 센서 측정값(하중, 변위)을 기반으로 재료의 공칭 응력(Engineering Stress, $\sigma$) 및 공칭 변형률(Engineering Strain, $\epsilon$) 계산.

### 3. 회귀 분석을 통한 탄성 계수(Young's Modulus) 추정 (`Scikit-Learn`)
- `Scikit-Learn`의 선형 회귀(Linear Regression) 모델을 적용하여 탄성 구간(Elastic Region)의 기울기를 정밀 추정 및 데이터화.

### 4. 시각화 및 결과 리포트 자동 내보내기 (Automated Reporting)
- 각 하중 조건별(예: 9k, 57k, 115k) 응력-변형률 선도 그래프(`.png`) 및 가공 데이터(`.csv`)를 결과 폴더에 자동 분류 저장.

---

## 📁 Repository Structure

```text
statics_analysis/
├── data/                           # [Input] LabVIEW (.lvm) 원본 측정 데이터
│   ├── 115k_*.lvm                  # - 115k 조건 데이터
│   ├── 57k_*.lvm                   # - 57k 조건 데이터
│   └── 9k_*.lvm                    # - 9k 조건 데이터
├── result/                         # [Output] 자동 생성 결과물
│   ├── *_stress_strain.png         # - 응력-변형률 선도 그래프
│   └── *_stress_strain_data.csv    # - 수치 가공 CSV 리포트
├── main.py                         # 데이터 파싱, 계산, 회귀 분석, 시각화 메인 스크립트
├── requirements.txt                # 개발 환경 의존성 목록
└── README.md                       # 프로젝트 포트폴리오 문서
