import os
import re
import io
import logging
import subprocess
import traceback
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import streamlit as st

# ------------------------------------------------------
# aac-metrics (MACE)
# ------------------------------------------------------
try:
    from aac_metrics.functional.mace import mace as mace_evaluate_func
    AAC_METRICS_AVAILABLE = True
except ImportError:
    AAC_METRICS_AVAILABLE = False
    mace_evaluate_func = None

# ------------------------------------------------------
# OpenAI 번역
# ------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ------------------------------------------------------
# dotenv
# ------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ------------------------------------------------------
# COMET-QE
# ------------------------------------------------------
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except Exception:
    COMET_AVAILABLE = False
    download_model = None
    load_from_checkpoint = None


# ======================================================
# Torchaudio load 패치 (soundfile 고정)
# ======================================================
def force_soundfile_load(filepath, **kwargs):
    wav, sr = sf.read(filepath)
    tensor = torch.from_numpy(wav).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.t()
    return tensor, sr

torchaudio.load = force_soundfile_load


# ======================================================
# 로깅
# ======================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ======================================================
# 설정
# ======================================================
AF3_INFERENCE_SCRIPT = "/workspace/audio-flamingo/llava/cli/infer_audio.py"
AUDIO_DIRECTORY = "/workspace/audio-flamingo/static/audio"

# MACE
MACE_THRESHOLD = 0.4

# COMET-QE 3-Step 임계값
THRESHOLD_CRITICAL = 0.35
THRESHOLD_RETRY = 0.35

# 번역 모델
TRANSLATION_MODEL = "gpt-4o-mini"
TRANSLATION_TEMPERATURE = 0.7

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
""".strip()


# ======================================================
# 키 안전화
# ======================================================
def safe_key(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", str(text))


# ======================================================
# COMET-QE 로딩
# ======================================================
@st.cache_resource(show_spinner=False)
def load_comet_model():
    if not COMET_AVAILABLE:
        return None
    try:
        model_name = "Unbabel/wmt20-comet-qe-da"
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        return model
    except Exception:
        logger.exception("COMET 모델 로딩 실패")
        return None


# ======================================================
# 더 좋은 번역만 반영
# ======================================================
def merge_better_results(df, old_col, old_score_col, new_col, new_score_col):
    for i in range(len(df)):
        if pd.notna(df.loc[i, new_score_col]):
            old_score = df.loc[i, old_score_col]
            new_score = df.loc[i, new_score_col]
            if pd.isna(old_score) or new_score > old_score:
                df.loc[i, old_col] = df.loc[i, new_col]
                df.loc[i, old_score_col] = new_score
    return df


# ======================================================
# GPT 번역 배치
# ======================================================
def _strip_numbering(text: str) -> str:
    return re.sub(r"^\d+\.\s*", "", str(text)).strip()


def _get_openai_key() -> Optional[str]:
    try:
        if hasattr(st, "secrets"):
            key = st.secrets.get("OPENAI_API_KEY", None)
            if key:
                return key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def translate_captions_in_batches(captions: List[str], batch_size=5, prompt_type="base"):
    captions = ["" if pd.isna(c) else str(c).strip() for c in captions]

    api_key = _get_openai_key()
    if not api_key or OpenAI is None:
        # 키 없으면 죽지 않고 스킵
        return [""] * len(captions)

    client = OpenAI(api_key=api_key)

    system_base = (
        "You are an expert translator specializing in English-to-Korean audio event descriptions. "
        "Translate faithfully, naturally, and concisely."
    )
    system_alt = (
        "You are a professional Korean linguist and sound description translator. "
        "Translate each caption into fluent, natural Korean that reflects the exact meaning, tone, and context."
    )

    total_translations: List[str] = []

    for i in range(0, len(captions), batch_size):
        window = captions[i:i + batch_size]
        batch = [cap for cap in window if cap]

        if not batch:
            total_translations.extend([""] * len(window))
            continue

        prompt_captions = "\n".join([f"{j + 1}. {text}" for j, text in enumerate(batch)])

        if prompt_type == "alt":
            user_prompt = f"""
Translate the following {len(batch)} English captions into Korean.
Each caption describes a sound or audio event.

[Input Captions]
{prompt_captions}

[Guidelines]
1. Write fluent, natural Korean as if written by a native sound expert.
2. Avoid literal or awkward phrasing.
3. Return ONLY a numbered list of Korean sentences.

[Korean Translations]
""".strip()
            system_prompt = system_alt
        else:
            user_prompt = f"""
Translate the following {len(batch)} English captions into Korean.

[Input Captions]
{prompt_captions}

[Rules]
1. Output ONLY a numbered list of Korean translations.
2. Match the count exactly.
3. No English, comments, or explanations.

[Korean Translations]
""".strip()
            system_prompt = system_base

        try:
            response = client.chat.completions.create(
                model=TRANSLATION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TRANSLATION_TEMPERATURE,
            )
            translated_text = (response.choices[0].message.content or "").strip()
            translated_batch = [
                _strip_numbering(line)
                for line in translated_text.split("\n")
                if line.strip()
            ]

            if len(translated_batch) != len(batch):
                if len(translated_batch) < len(batch):
                    translated_batch.extend([""] * (len(batch) - len(translated_batch)))
                else:
                    translated_batch = translated_batch[:len(batch)]

            final_batch: List[str] = []
            t_idx = 0
            for cap in window:
                if cap:
                    final_batch.append(translated_batch[t_idx] if t_idx < len(translated_batch) else "")
                    t_idx += 1
                else:
                    final_batch.append("")
            total_translations.extend(final_batch)

        except Exception:
            logger.exception("번역 오류")
            total_translations.extend([""] * len(window))

    return total_translations


# ======================================================
# COMET-QE 점수 계산
# ======================================================
def evaluate_translations(model, captions, translations):
    if model is None:
        return [np.nan] * len(captions)

    data = []
    valid_positions = []
    for i, (src, mt) in enumerate(zip(captions, translations)):
        src = "" if src is None else str(src).strip()
        mt = "" if mt is None else str(mt).strip()
        if src and mt:
            data.append({"src": src, "mt": mt})
            valid_positions.append(i)

    if not data:
        return [np.nan] * len(captions)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpus = 1 if device == "cuda" else 0

    try:
        out = model.predict(data, batch_size=32, gpus=gpus, progress_bar=False)
        scores = out.scores
    except Exception:
        logger.exception("COMET-QE predict 실패")
        return [np.nan] * len(captions)

    result = [np.nan] * len(captions)
    for s, pos in zip(scores, valid_positions):
        result[pos] = float(s)

    return result


# ======================================================
# 번역 3-Step QE 로직
# ======================================================
def translate_captions_with_qe_3step(caption_list: List[str], comet_model):
    df = pd.DataFrame({"caption": caption_list})

    # Step 1) Base
    df["translated"] = translate_captions_in_batches(
        df["caption"].tolist(),
        batch_size=max(2, len(df)),
        prompt_type="base"
    )
    df["translated_qe_score"] = evaluate_translations(
        comet_model,
        df["caption"].tolist(),
        df["translated"].tolist()
    )

    if comet_model is None:
        return df

    # Step 2) Critical 미만 -> Alt
    mask_critical = (df["translated_qe_score"] < THRESHOLD_CRITICAL) & df["translated_qe_score"].notna()
    if mask_critical.any():
        idxs = df[mask_critical].index.tolist()

        alt_trans = translate_captions_in_batches(
            df.loc[idxs, "caption"].tolist(),
            batch_size=max(2, len(idxs)),
            prompt_type="alt"
        )
        alt_scores = evaluate_translations(
            comet_model,
            df.loc[idxs, "caption"].tolist(),
            alt_trans
        )

        df.loc[idxs, "temp_translated"] = alt_trans
        df.loc[idxs, "temp_score"] = alt_scores
        df = merge_better_results(df, "translated", "translated_qe_score", "temp_translated", "temp_score")
        df.drop(columns=["temp_translated", "temp_score"], inplace=True, errors="ignore")

    # Step 3) Retry 미만 -> Base 재시도
    mask_retry = (df["translated_qe_score"] < THRESHOLD_RETRY) & df["translated_qe_score"].notna()
    if mask_retry.any():
        idxs = df[mask_retry].index.tolist()

        re_trans = translate_captions_in_batches(
            df.loc[idxs, "caption"].tolist(),
            batch_size=max(2, len(idxs)),
            prompt_type="base"
        )
        re_scores = evaluate_translations(
            comet_model,
            df.loc[idxs, "caption"].tolist(),
            re_trans
        )

        df.loc[idxs, "temp_translated"] = re_trans
        df.loc[idxs, "temp_score"] = re_scores
        df = merge_better_results(df, "translated", "translated_qe_score", "temp_translated", "temp_score")
        df.drop(columns=["temp_translated", "temp_score"], inplace=True, errors="ignore")

    return df


# ======================================================
# MACE warmup
# ======================================================
def initialize_mace_model(device: str) -> bool:
    if not AAC_METRICS_AVAILABLE:
        return False
    try:
        candidates = ["a dog is barking", "a car is passing by"]
        references = [["dummy"], ["dummy"]]
        dummy_audio_path = "warmup_dummy_audio.wav"
        sf.write(dummy_audio_path, np.zeros(32000), 32000)

        _ = mace_evaluate_func(
            candidates=candidates,
            mult_references=references,
            audio_paths=[dummy_audio_path, dummy_audio_path],
            device=device,
            mace_method="audio"
        )
        os.remove(dummy_audio_path)
        return True
    except Exception:
        logger.exception("MACE 모델 초기화 실패")
        return False


# ======================================================
# AF3 호출 (conda 실패 시 python fallback)
# ======================================================
def _af3_command_conda(audio_path: str, prompt: str):
    return [
        "conda", "run", "-n", "af3", "python",
        AF3_INFERENCE_SCRIPT,
        "--model-base", "nvidia/audio-flamingo-3",
        "--conv-mode", "auto",
        "--text", prompt,
        "--media", audio_path
    ]


def _af3_command_python(audio_path: str, prompt: str):
    return [
        "python",
        AF3_INFERENCE_SCRIPT,
        "--model-base", "nvidia/audio-flamingo-3",
        "--conv-mode", "auto",
        "--text", prompt,
        "--media", audio_path
    ]


def call_af3_subprocess(audio_path: str, prompt: str) -> Optional[str]:
    commands = [
        _af3_command_conda(audio_path, prompt),
        _af3_command_python(audio_path, prompt),
    ]

    last_err = None
    for cmd in commands:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            output = result.stdout.strip()
            if output:
                lines = [l.strip() for l in output.splitlines() if l.strip()]
                return lines[-1] if lines else None
            return None
        except Exception as e:
            last_err = e

    logger.exception("AF3 subprocess 실패", exc_info=last_err)
    return None


# ======================================================
# MACE 평가
# ======================================================
def evaluate_caption_with_mace(candidate_caption: str, audio_path: str, device: str) -> Optional[float]:
    if not AAC_METRICS_AVAILABLE:
        return None
    dummy_reference = [["dummy"]]
    try:
        corpus_scores, _ = mace_evaluate_func(
            candidates=[candidate_caption],
            mult_references=dummy_reference,
            audio_paths=[audio_path],
            device=device,
            mace_method="audio"
        )
        return corpus_scores["mace"].item()
    except Exception:
        logger.exception("MACE 평가 실패")
        return None


# ======================================================
# 캡션 생성 (영어 1개)
# ======================================================
def get_one_good_caption(audio_path: str, device: str):
    caption = call_af3_subprocess(audio_path, AF3_PROMPT)
    if caption is None:
        return None, "API_FAIL_1"

    score = evaluate_caption_with_mace(caption, audio_path, device)
    if score is None:
        return caption, "MACE_SKIPPED_OR_FAIL"

    if score >= MACE_THRESHOLD:
        return caption, f"PASSED (Score: {score:.4f})"

    caption_retry = call_af3_subprocess(audio_path, AF3_PROMPT)
    if caption_retry is None:
        return None, "FILTER_FAILED_RETRY_API"

    score_retry = evaluate_caption_with_mace(caption_retry, audio_path, device)
    if score_retry and score_retry >= MACE_THRESHOLD:
        return caption_retry, f"REGEN_PASSED (Score: {score_retry:.4f})"

    return None, f"REGEN_FAILED (Score: {score_retry:.4f})" if score_retry is not None else "REGEN_FAILED"


# ======================================================
# 업로드 파일을 static/audio에 저장/재사용
# ======================================================
def ensure_audio_in_static_dir(uploaded_file, overwrite: bool = False) -> str:
    os.makedirs(AUDIO_DIRECTORY, exist_ok=True)
    base_name = os.path.basename(uploaded_file.name)
    target_path = os.path.join(AUDIO_DIRECTORY, base_name)

    if (not os.path.exists(target_path)) or overwrite:
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return target_path


# ======================================================
# 단일 오디오 처리 (영문 3개 생성 + KO 번역 필터)
# ======================================================
def run_for_single_audio(audio_path: str, comet_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    caps_en: List[str] = []
    statuses: List[str] = []

    for _ in range(3):
        c_en, s = get_one_good_caption(audio_path, device)
        caps_en.append(c_en or "")
        statuses.append(s)

    df_t = translate_captions_with_qe_3step(caps_en, comet_model)

    caps_ko = df_t["translated"].tolist() if "translated" in df_t.columns else [""] * 3
    qe_scores = df_t["translated_qe_score"].tolist() if "translated_qe_score" in df_t.columns else [np.nan] * 3

    while len(caps_ko) < 3:
        caps_ko.append("")
    while len(qe_scores) < 3:
        qe_scores.append(np.nan)

    return {
        "caps_en": caps_en[:3],   # 내부용
        "caps_ko": caps_ko[:3],   # UI/엑셀
        "statuses": statuses[:3],
        "qe_scores": qe_scores[:3],
    }


# ======================================================
# Streamlit UI
#  - 표시 최소화:
#    1) 결과 테이블
#    2) 수정/선택 UI
# ======================================================
def main():
    st.set_page_config(page_title="AF3 Caption Batch (KO only)", layout="wide")
    st.title("🎧 AF3 한국어 캡션 생성 벤치마크")

    # -----------------------------
    # 세션 상태 초기화
    # -----------------------------
    if "run_requested" not in st.session_state:
        st.session_state["run_requested"] = False
    if "is_running" not in st.session_state:
        st.session_state["is_running"] = False

    # -----------------------------
    # 설정(사이드바)
    # -----------------------------
    with st.sidebar:
        st.markdown("### 설정")

        global MACE_THRESHOLD, THRESHOLD_CRITICAL, THRESHOLD_RETRY
        global TRANSLATION_MODEL, TRANSLATION_TEMPERATURE

        MACE_THRESHOLD = st.slider("MACE Threshold", 0.0, 1.0, float(MACE_THRESHOLD), 0.01)
        THRESHOLD_CRITICAL = st.slider("Critical threshold (Alt)", 0.0, 1.0, float(THRESHOLD_CRITICAL), 0.01)
        THRESHOLD_RETRY = st.slider("Retry threshold (Base)", 0.0, 1.0, float(THRESHOLD_RETRY), 0.01)

        TRANSLATION_MODEL = st.text_input("Translation model", TRANSLATION_MODEL)
        TRANSLATION_TEMPERATURE = st.slider("Translation temperature", 0.0, 1.0, float(TRANSLATION_TEMPERATURE), 0.05)

        st.markdown("---")
        overwrite = st.checkbox("같은 파일명 덮어쓰기", value=False)

    # -----------------------------
    # 입력
    # -----------------------------
    uploaded_list = st.file_uploader(
        "오디오 파일 업로드",
        type=["wav", "flac", "mp3", "ogg"],
        accept_multiple_files=True
    )

    # 버튼은 "요청 플래그"만 켠다
    if st.button("배치 캡션 생성", type="primary", disabled=st.session_state["is_running"]):
        st.session_state["run_requested"] = True

    # -----------------------------
    # 처리 블록: run_requested가 True일 때만
    # -----------------------------
    if st.session_state["run_requested"] and not st.session_state["is_running"]:
        st.session_state["is_running"] = True

        try:
            if not uploaded_list:
                st.warning("파일을 업로드하세요.")
                st.session_state["run_requested"] = False
                st.session_state["is_running"] = False
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"

                if "mace_warmed" not in st.session_state:
                    with st.spinner("MACE 모델 예열 중..."):
                        st.session_state["mace_warmed"] = initialize_mace_model(device)

                with st.spinner("COMET-QE 모델 로딩 중..."):
                    comet_model = load_comet_model()

                # ✅ placeholder를 한 번만 만들고 재사용
                status_box = st.empty()
                progress = st.progress(0)

                results: List[Dict[str, Any]] = []
                total = len(uploaded_list)

                for idx, uploaded in enumerate(uploaded_list, start=1):
                    status_box.info(f"[{idx}/{total}] {uploaded.name}")

                    audio_path = ensure_audio_in_static_dir(uploaded, overwrite=overwrite)
                    res = run_for_single_audio(audio_path, comet_model)

                    results.append({
                        "file": os.path.basename(audio_path),
                        "cap1_ko": res["caps_ko"][0],
                        "cap2_ko": res["caps_ko"][1],
                        "cap3_ko": res["caps_ko"][2],
                        "status1": res["statuses"][0],
                        "status2": res["statuses"][1],
                        "status3": res["statuses"][2],
                        "qe1": res["qe_scores"][0],
                        "qe2": res["qe_scores"][1],
                        "qe3": res["qe_scores"][2],
                    })

                    progress.progress(int(idx / total * 100))

                status_box.empty()
                progress.empty()

                st.session_state["df_out"] = pd.DataFrame(results)

                # ✅ 완료 후 플래그 해제
                st.session_state["run_requested"] = False
                st.session_state["is_running"] = False

        except Exception:
            st.session_state["run_requested"] = False
            st.session_state["is_running"] = False
            st.error("배치 처리 중 오류가 발생했습니다.")
            st.exception(traceback.format_exc())
            return

    # -----------------------------
    # 결과/수정 UI
    # -----------------------------
    df_out = st.session_state.get("df_out", None)
    if df_out is None or not isinstance(df_out, pd.DataFrame) or df_out.empty:
        return

    # 1) 결과 테이블 (KO 3개)
    st.dataframe(df_out, use_container_width=True)

    # 2) 수정/선택 UI
    edited_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []

    for _, r in df_out.iterrows():
        fname = r["file"]
        safe_fname = safe_key(fname)
        key_prefix = f"capedit::{safe_fname}"

        with st.expander(f"{fname}", expanded=True):
            cols = st.columns(3)

            for j in [1, 2, 3]:
                k_ko = f"{key_prefix}::ko{j}"
                if k_ko not in st.session_state:
                    st.session_state[k_ko] = r.get(f"cap{j}_ko", "") or ""

            for j, col in zip([1, 2, 3], cols):
                with col:
                    st.text_area(
                        f"Caption {j} (KO)",
                        key=f"{key_prefix}::ko{j}",
                        height=110
                    )

            sel_key = f"{key_prefix}::selected"
            if sel_key not in st.session_state:
                st.session_state[sel_key] = "1"

            st.radio(
                "엑셀에 저장할 캡션 선택",
                options=["1", "2", "3"],
                horizontal=True,
                key=sel_key
            )

        edited_row = {
            "file": fname,
            "cap1_ko": st.session_state.get(f"{key_prefix}::ko1", ""),
            "cap2_ko": st.session_state.get(f"{key_prefix}::ko2", ""),
            "cap3_ko": st.session_state.get(f"{key_prefix}::ko3", ""),
            "status1": r.get("status1", ""),
            "status2": r.get("status2", ""),
            "status3": r.get("status3", ""),
            "qe1": r.get("qe1", np.nan),
            "qe2": r.get("qe2", np.nan),
            "qe3": r.get("qe3", np.nan),
            "selected_caption": st.session_state.get(sel_key, "1"),
        }
        edited_rows.append(edited_row)

        sidx = int(edited_row["selected_caption"])
        selected_rows.append({
            "file": fname,
            "selected_idx": sidx,
            "caption_ko": edited_row.get(f"cap{sidx}_ko", ""),
            "qe_score": edited_row.get(f"qe{sidx}", np.nan),
            "status": edited_row.get(f"status{sidx}", ""),
        })

    df_edited = pd.DataFrame(edited_rows)
    df_selected = pd.DataFrame(selected_rows)

    buffer_all = io.BytesIO()
    df_edited.to_excel(buffer_all, index=False)
    st.download_button(
        label="전체 3캡션(KO) 엑셀",
        data=buffer_all.getvalue(),
        file_name="af3_caption_batch_3caps_KO_edited.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    buffer_sel = io.BytesIO()
    df_selected.to_excel(buffer_sel, index=False)
    st.download_button(
        label="선택 1캡션(KO) 엑셀",
        data=buffer_sel.getvalue(),
        file_name="af3_caption_batch_selected_1cap_KO.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


if __name__ == "__main__":
    main()

