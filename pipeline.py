import torch
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import soundfile as sf
import logging
import sys
import subprocess
import json
from typing import List, Optional, Tuple
import glob
import traceback
import torchaudio  # <--- [필수 추가] 패치를 위해 필요

# ======================================================
# 🛠️ [핵심] Torchaudio 강제 패치 (이 부분만 추가됨)
# 설명: 시스템 충돌을 막기 위해 soundfile을 강제로 사용하게 설정
# ======================================================
def force_soundfile_load(filepath, **kwargs):
    wav, sr = sf.read(filepath)
    tensor = torch.from_numpy(wav).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.t()
    return tensor, sr

# 패치 적용
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
torchaudio.load = force_soundfile_load
logger.info("✅ Torchaudio patched to use soundfile.")
# ======================================================


# --- aac-metrics 임포트 ---
try:
    from aac_metrics.functional.mace import mace as mace_evaluate_func
    AAC_METRICS_AVAILABLE = True
except ImportError:
    AAC_METRICS_AVAILABLE = False
    mace_evaluate_func = None

AF3_INFERENCE_SCRIPT = "/workspace/audio-flamingo/llava/cli/infer_audio.py"
AF3_PROMPT = """
You are “Audio Analyst.” 
You have to analyse audio and make caption.
Please think and reason about the input audio before you respond.

Your respond SHOULD follow under structure for each audio:
    "caption": "<exaplain1 about audio> and <explain2 about aduio>"

respond Examples : 
respond example 1 :
    "caption": "<exaplain1 about audio> and <explain2 about aduio> while <action1>" 

respond example 2 :  
    "caption": "<exaplain1 about audio>"
   
Key Requirements :
- respond SHOULD be one sentence, 8–20 words.
- respond SHOULD be Present tense, objective, neutral, SHOULD NOT narrative or speculation.
- SHOULD NOT use brands, proper names, quoted phrases, narrative text, or invented causes.
- SHOULD NOT use unclear words(e.g., Do not use "buzzes", "chirp", "rumble", "revs", "idles", "hum", "roaring")
    - You SHOULD NOT copy examples
- SHOULD NOT describe unheard sound.
"""
AUDIO_DIRECTORY = "/workspace/audio-flamingo/static/audio"
MACE_THRESHOLD = 0.4

def initialize_mace_model(device: str) -> bool:
    if not AAC_METRICS_AVAILABLE:
        return False
    try:
        logger.info("⏳ MACE 모델 예열 중...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Mace DEVICE: {device}")

        candidates = ["a dog is barking", "a car is passing by"]
        references = [["dummy"], ["dummy"]]  # dummy GT
        dummy_audio_path = "warmup_dummy_audio.wav"
        sf.write(dummy_audio_path, np.zeros(32000), 32000)

        _ = mace_evaluate_func(
            candidates=candidates,
            mult_references=references,
            audio_paths=[dummy_audio_path, dummy_audio_path],
            device=device,
            mace_method="audio"
        )

        logger.info("✅ MACE 모델 준비 완료!")
        os.remove(dummy_audio_path)
        return True

    except Exception as e:
        print("\n🔥 ERROR: MACE warm-up failure")
        traceback.print_exc()
        logger.error("❌ MACE 모델 초기화 실패", exc_info=True)
        return False


# --- AF3 호출 ---
def call_af3_subprocess(audio_path: str, prompt: str) -> Optional[str]:
    command = [
        "conda", "run", "-n", "af3", "python",
        AF3_INFERENCE_SCRIPT,
        "--model-base", "nvidia/audio-flamingo-3",
        "--conv-mode", "auto",
        "--text", prompt,
        "--media", audio_path
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
        
        #caption 생성 결과 출력 후 parse.
        #이를 3개 생성하여 원하는 겨로가 출력하도록 진행
        #출력 시 temperature을 변경하는 건 어떤가?
        output = result.stdout.strip()

        if output:
            lines = [l.strip() for l in output.splitlines() if l.strip()]
            return lines[-1] if lines else None
        return None

    except Exception as e:
        print("\n🔥 ERROR: AF3 subprocess execution failure")
        print("Command:", " ".join(command))
        traceback.print_exc()
        logger.error("af3 실행 오류", exc_info=True)
        return None


# --- MACE 평가 ---
def evaluate_caption_with_mace(candidate_caption: str, audio_path: str, device: str) -> Optional[float]:
    dummy_reference = [["dummy"]]  # 반드시 2D list

    try:
        corpus_scores, _ = mace_evaluate_func(
            candidates=[candidate_caption],
            mult_references=dummy_reference,  # None 완전 제거
            audio_paths=[audio_path],
            device=device,
            mace_method="audio"
        )
        score = corpus_scores["mace"].item()
        return score

    except Exception as e:
        print("\n🔥 ERROR: MACE evaluation failure")
        print("caption:", candidate_caption)
        print("audio:", audio_path)
        traceback.print_exc()
        logger.error("MACE 평가 오류", exc_info=True)
        return None


# --- 캡션 생성 ---
def get_one_good_caption(audio_path: str, device: str):
    caption = call_af3_subprocess(audio_path, AF3_PROMPT)
    if caption is None:
        return None, "API_FAIL_1"

    score = evaluate_caption_with_mace(caption, audio_path, device)
    if score is None:
        return caption, "MACE_FAIL"

    if score >= MACE_THRESHOLD:
        return caption, f"PASSED (Score: {score:.4f})"

    caption_retry = call_af3_subprocess(audio_path, AF3_PROMPT)
    if caption_retry is None:
        return None, "FILTER_FAILED_RETRY_API"

    score_retry = evaluate_caption_with_mace(caption_retry, audio_path, device)
    if score_retry and score_retry >= MACE_THRESHOLD:
        return caption_retry, f"REGEN_PASSED (Score: {score_retry:.4f})"

    return None, f"REGEN_FAILED (Score: {score_retry:.4f})"


def main_pipeline(audio_dir=AUDIO_DIRECTORY, output_excel="pipeline_output_final_2_captions.xlsx"):
    logger.info("🚀 Audio-only MACE filtering pipeline 시작")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"DEVICE: {device}")

    if not initialize_mace_model(device):
        return

    audio_files = []
    for ext in ["*.wav", "*.flac", "*.mp3", "*.ogg"]:
        audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))

    results = []
    for audio_path in tqdm(audio_files, desc="Processing"):
        c1, s1 = get_one_good_caption(audio_path, device)
        c2, s2 = get_one_good_caption(audio_path, device)

        results.append({"file": os.path.basename(audio_path), "cap1": c1, "cap2": c2, "status1": s1, "status2": s2})

    pd.DataFrame(results).to_excel(output_excel, index=False)
    logger.info("🎉 완료!")

    #임시 체크 용 
    #print(c1)


if __name__ == "__main__":
    main_pipeline()