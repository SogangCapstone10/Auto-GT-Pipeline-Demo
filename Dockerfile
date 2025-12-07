# 1. 'audio-flamingo'의 cu128과 호환되는 CUDA 12.1.1 이미지를 사용
# (이것으로 1.8GB짜리 불필요한 conda cuda-toolkit 다운로드 문제를 해결)
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# 2. 작업 디렉토리 설정
WORKDIR /workspace/

# 3. 빌드에 필요한 시스템 도구 통합 설치
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    pkg-config \
    nano \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 4. Miniconda 설치
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# 5. Conda 실행 파일을 시스템 PATH에 추가
ENV PATH="/opt/conda/bin:${PATH}"

# 6. 'clair-a'를 포함한 모든 리포지토리 복제
# (KoSimCSE 폴더 이름을 'KoSimCSE-SKT'로 지정)
RUN git clone --depth 1 -b audio_flamingo_3 https://github.com/SogangCapstone10/audio-flamingo.git && \
    git clone https://github.com/SogangCapstone10/clair-a.git && \
    git clone https://github.com/SogangCapstone10/KoSimCSE-SKT.git && \
    git clone https://github.com/SogangCapstone10/KoBERT.git KoSimCSE-SKT/KoBERT

# --- 환경 1: Audio Flamingo (af3) 격리 설치 (모든 오류 수정) ---
# 7. Audio Flamingo 환경 설정 스크립트 '수정' 및 실행
RUN cd audio-flamingo && \
    \
    # (수정 1) 1.8GB 다운로드 유발하는 'conda install cuda-toolkit' 라인을 주석 처리(#)
    sed -i 's/conda install -c nvidia cuda-toolkit -y/# conda install -c nvidia cuda-toolkit -y/' environment_setup.sh && \
    \
    # (수정 2) 'flash_attn' 빌드 오류 해결을 위해 --no-build-isolation 플래그 추가
    sed -i 's/pip install flash_attn==2.7.3/pip install --no-build-isolation flash_attn==2.7.3/' environment_setup.sh && \
    \
    # '수정된' 스크립트를 실행하여 af3 환경 생성
    bash environment_setup.sh af3

# --- 환경 2: Clair-A 및 모든 추가 도구 (clair_a_env) 설치 ---
# 8. 'clair_a_env' Conda 환경 생성
RUN conda create -n clair_a_env python=3.10 -y

# 9. 'clair_a_env'에 모든 추가 패키지 설치
# 셸을 'clair_a_env'로 변경
SHELL ["conda", "run", "-n", "clair_a_env", "/bin/bash", "-c"]

# 10. 모든 패키지 설치 (Rust 컴파일 오류 및 경로 수정 완료)
RUN \
    # Conda 패키지 설치 (flask)
    conda install -y flask && \
    \
    # clair-a 및 API 서버 관련
    pip install /workspace/clair-a/. && \
    pip install openai fastapi "uvicorn[standard]" && \
    \
    # aac-metrics 설치
    pip install aac-metrics && \
    \
    # KoBERT 설치 (경로 수정: KoSimCSE -> KoSimCSE-SKT)
    cd /workspace/KoSimCSE-SKT/KoBERT && \
    pip install -r requirements.txt && \
    pip install . && \
    \
    # KoSimCSE 설치 (경로 수정: KoSimCSE -> KoSimCSE-SKT)
    cd /workspace/KoSimCSE-SKT && \
    \
    # (*** Rust 오류 해결 ***)
    # 'transformers==2.8.0' 라인을 requirements.txt에서 강제 삭제
    sed -i '/transformers/d' requirements.txt && \
    \
    # 수정된 requirements.txt로 설치
    pip install -r requirements.txt && \
    \
    # 기타 모든 pip 패키지 (openpyxl 및 최신 transformers 포함)
    pip install pandas "unbabel-comet>=2.0" openpyxl tqdm transformers sentencepiece

# 11. aac-metrics용 추가 데이터 다운로드
RUN aac-metrics-download

# 12. 셸을 기본값으로 복원
SHELL ["/bin/bash", "-c"]

# 13. 최종 작업 디렉토리 설정
WORKDIR /workspace/

# 14. 컨테이너 시작 시 실행할 기본 명령어 (bash 셸)
CMD ["/bin/bash"]