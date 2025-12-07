# 🛠️ Auto-GT-Pipeline & Evaluation Framework

<div align="center">
  <img src="https://img.shields.io/badge/Docker-Environment-blue?logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/Python-3.10-yellow?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/AF3-LALM-red" alt="AF3">
  <img src="https://img.shields.io/badge/MACE-Filtering-green" alt="MACE">
  <img src="https://img.shields.io/badge/CLAIR--A-Judge-purple" alt="CLAIR-A">
</div>

<br/>

## 📌 소개 (Introduction)

이 레포지토리는 **LALM(Audio Flamingo 3)과 LLM(GPT-4o)을 결합한 오디오 캡션 자동 생성 파이프라인**과, 이를 통해 구축된 데이터셋을 기반으로 모델 성능을 정량적으로 검증하는 **벤치마크 평가 시스템**입니다.

복잡한 의존성(CUDA 버전, AF3 가상환경, Metric 라이브러리 충돌 등)을 해결하기 위해 **Docker** 기반의 통합 환경을 제공합니다. 사용자는 이 환경 내에서 데이터셋 구축(Generation)부터 모델 평가(Evaluation)까지 한번에 수행할 수 있습니다.

---

## 📂 핵심 파일 구성 (Key Components)

이 프로젝트는 4가지 핵심 파일로 구성되어 있으며, 각각 환경 구축, 데이터 생성, 평가, UI 연결을 담당합니다.

| 파일명 | 분류 | 상세 설명 |
|:---:|:---:|---|
| **`Dockerfile`** | **Environment** | **환경 구축 정의**<br>AF3(LALM), MACE(Filtering), CLAIR-A, KoBERT 등 필요한 모든 라이브러리를 포함합니다. CUDA 12.1.1 호환성 및 `torchaudio` 충돌 문제를 해결한 최종 이미지입니다. |
| **`pipeline.py`** | **Generation** | **GT 자동 생성 파이프라인 (CLI)**<br>오디오 폴더를 입력받아 `AF3 캡션 생성` → `MACE 필터링(0.4 Threshold)` → `자동 재시도` 로직을 수행합니다. UI 없이 대량의 오디오로부터 고품질 영어 캡션 데이터를 구축할 때 사용합니다. |
| **`evaluation_model.py`** | **Evaluation** | **벤치마크 성능 평가 (CLI)**<br>구축된 데이터셋(User GT)을 Ground Truth로 삼아 타 모델(WavCaps, Whisper 등)의 성능을 평가합니다. `SPIDEr-FL`, `FENSE`, `CLAIR-A` 지표를 산출하여 비교 분석합니다. |
| **`Ui_pipeline.py`** | **UI Backend** | **UI 시연 시스템 연동 모듈**<br>[UI Repository](https://github.com/SogangCapstone10/UI)의 Streamlit 인터페이스와 연결되는 백엔드 로직입니다. 웹에서 오디오를 업로드하고 생성/번역/필터링 과정을 시각적으로 제어할 수 있습니다. |

---

## 🐳 환경 구축 (Docker Setup)

이 프로젝트는 `base` 환경과 `af3` 환경이 공존하는 특수한 구조이므로, 반드시 제공된 Dockerfile을 사용해야 합니다.

```bash
# Dockerfile 위치에서 이미지 빌드
docker build -t auto-gt-pipeline:latest .

# GPU 활성화 + OpenAI API 키 포함하여 컨테이너 실행
docker run -it --gpus all \
  -v $(pwd)/data:/workspace/data \
  -e OPENAI_API_KEY="sk-proj-..." \
  auto-gt-pipeline:latest
```
## 🚀 사용 가이드 (Usage Scenarios)

### Scenario A: 데이터셋 대량 자동 구축 (Headless)

UI 없이 폴더 내 모든 오디오 파일에 대해 캡션을 생성하고 필터링하려면 `pipeline.py`를 사용합니다.

```bash
# AUDIO_DIRECTORY 경로를 pipeline.py에서 수정 후 실행
python pipeline.py
```
### Scenario B: 모델 벤치마크 평가 (Evaluation)

```bash
# 비교 대상 모델 결과 파일 준비 
python evaluation_model.py
```
### Scenario C: UI 기반 시연 (Interactive Demo)
웹 인터페이스를 통해 시스템을 시연하거나 데이터를 선별하려면 Ui_pipeline.py를 실행합니다.

```bash
streamlit run Ui_pipeline.py

```
