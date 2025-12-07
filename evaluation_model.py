import os
import sys
import pandas as pd
import numpy as np
import logging
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# aac-metrics
from aac_metrics import evaluate as aac_evaluate

# --- 1. Configuration ---
GT_CLOTHO_PATH = "clotho_captions_development.csv"
GT_USER_PATH = "final_comprehensive_summary.xlsx"
OUTPUT_EXCEL_FILE = "final_evaluation_comparison.xlsx"

CANDIDATE_FILES = {
    "WavCaps": "wavcaps_captions_result.xlsx",
    "Qwen": "qwen_captions_result.xlsx",
    "Whisper": "whisper_captions_result.xlsx"
}

# 'fense'를 계산하면 'sbert_sim'도 sents_scores에 자동으로 포함됨
METRICS_AAC = ["spider_fl", "fense"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPENAI_MODEL = 'openai/gpt-5' 
USER_GT_THRESHOLD = 0.8

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Library Setup ---
clair_lib_path = '/workspace/clair-a'
if clair_lib_path not in sys.path: sys.path.append(clair_lib_path)
try:
    from clair_a import clair_a
except ImportError:
    logger.critical("❌ 'clair_a' import failed.")
    sys.exit(1)

excel_summary_data = []
excel_distribution_data = []

# --- Helper Functions ---

def load_filtered_data():
    try:
        df_user = pd.read_excel(GT_USER_PATH)
        df_user['key'] = df_user['file_name'].apply(lambda x: os.path.basename(str(x)).strip())
        if 'clair_score' in df_user.columns:
            df_user = df_user[df_user['clair_score'] >= USER_GT_THRESHOLD].copy()
        else:
            logger.critical("❌ 'clair_score' column missing.")
            sys.exit(1)
        valid_keys = set(df_user['key'].tolist())
        df_user = df_user.rename(columns={'generated_caption': 'user_gt_caption'}).set_index('key')
        logger.info(f"🧹 User GT Filtered: {len(df_user)} files (Score >= {USER_GT_THRESHOLD})")
    except Exception as e:
        logger.critical(f"Failed to load User GT: {e}")
        sys.exit(1)

    try:
        df_clotho = pd.read_csv(GT_CLOTHO_PATH)
        df_clotho['key'] = df_clotho['file_name'].apply(lambda x: os.path.basename(str(x)).strip())
        df_clotho = df_clotho[df_clotho['key'].isin(valid_keys)].copy().set_index('key')
        logger.info(f"🧹 Clotho GT Filtered: {len(df_clotho)} files matching User GT")
    except Exception as e:
        logger.critical(f"Failed to load Clotho GT: {e}")
        sys.exit(1)

    return df_clotho, df_user, valid_keys

def calculate_distribution(scores, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0]):
    scores = np.array(scores)
    scores = scores[~np.isnan(scores)]
    if len(scores) == 0: return {}, 0.0
    hist, _ = np.histogram(scores, bins=bins)
    percentages = (hist / len(scores)) * 100
    dist_dict = {}
    for i in range(len(bins)-1):
        key = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
        dist_dict[key] = percentages[i]
    return dist_dict, np.mean(scores)

def calculate_clair_parallel(candidates, references, model_name):
    def _run_single(args):
        cand, refs = args
        if not cand: return 0.0
        try:
            score, _ = clair_a(candidate=cand, targets=refs, model=model_name, tiebreaking_method="fense")
            return score
        except: return 0.0
    tasks = list(zip(candidates, references))
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(tqdm(executor.map(_run_single, tasks), total=len(tasks), desc="CLAIR-A API", leave=False))
    return results

def process_metric(model_name, metric_name, scores_a, scores_b, bins):
    # None 체크
    if scores_a is None or scores_b is None:
        logger.warning(f"⚠️ Skipping {metric_name} because scores are missing (None).")
        return
    
    # 빈 리스트 체크
    if len(scores_a) == 0 or len(scores_b) == 0:
        logger.warning(f"⚠️ Skipping {metric_name} because scores list is empty.")
        return

    dist_a, mean_a = calculate_distribution(scores_a, bins)
    dist_b, mean_b = calculate_distribution(scores_b, bins)
    
    correlation = 0.0
    if len(scores_a) > 1:
        try:
            correlation = np.corrcoef(scores_a, scores_b)[0,1]
        except:
            correlation = 0.0
    
    excel_summary_data.append({
        "Model": model_name,
        "Metric": metric_name,
        "Mean_Scenario_A (Clotho)": mean_a,
        "Mean_Scenario_B (User)": mean_b,
        "Mean_Diff": abs(mean_a - mean_b),
        "Correlation": correlation
    })
    
    for range_key in dist_a.keys():
        excel_distribution_data.append({
            "Model": model_name,
            "Metric": metric_name,
            "Range": range_key,
            "Pct_Scenario_A (Clotho)": dist_a[range_key],
            "Pct_Scenario_B (User)": dist_b[range_key],
            "Pct_Diff": abs(dist_a[range_key] - dist_b[range_key])
        })
    print(f"   [{metric_name}] Corr: {correlation:.3f} | Mean Diff: {abs(mean_a - mean_b):.4f}")

def run_evaluation(model_name, candidate_path, df_clotho, df_user):
    logger.info(f"🚀 Evaluating Model: {model_name}")
    try:
        df_cand = pd.read_excel(candidate_path)
        df_cand['key'] = df_cand['audio_file'].apply(lambda x: os.path.basename(str(x)).strip())
    except: return

    common_keys = list(set(df_cand['key']) & set(df_clotho.index) & set(df_user.index))
    if not common_keys: return

    df_cand = df_cand.set_index('key').reindex(common_keys)
    df_clotho_s = df_clotho.reindex(common_keys)
    df_user_s = df_user.reindex(common_keys)
    
    candidates = df_cand['generated_caption'].astype(str).tolist()
    refs_clotho = [[str(r[c]) for c in ['caption_1','caption_2','caption_3','caption_4','caption_5'] if pd.notna(r[c])] for _, r in df_clotho_s.iterrows()]
    refs_user = [[str(cap)] for cap in df_user_s['user_gt_caption']]

    logger.info("  > Calculating AAC Metrics (SPIDEr-FL, FENSE)...")
    
    try:
        # ❗ [핵심 수정] _, sc_a = ... (첫번째 리턴값은 평균이므로 버리고, 두번째인 리스트를 취함)
        _, sc_a = aac_evaluate(candidates=candidates, mult_references=refs_clotho, metrics=METRICS_AAC, device=DEVICE, verbose=0)
        _, sc_b = aac_evaluate(candidates=candidates, mult_references=refs_user, metrics=METRICS_AAC, device=DEVICE, verbose=0)
        
    except Exception as e:
        logger.error(f"❌ AAC Evaluate Error: {e}")
        sc_a, sc_b = {}, {}

    logger.info("  > Calculating CLAIR-A...")
    cl_a = calculate_clair_parallel(candidates, refs_clotho, OPENAI_MODEL)
    cl_b = calculate_clair_parallel(candidates, refs_user, OPENAI_MODEL)

    standard_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    spider_bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]

    # 1. SBERT Sim
    if 'sbert_sim' in sc_a and 'sbert_sim' in sc_b:
        process_metric(model_name, "SBERT_Sim", sc_a['sbert_sim'].tolist(), sc_b['sbert_sim'].tolist(), standard_bins)
    else:
        logger.warning(f"⚠️ 'sbert_sim' key not found. Keys returned: {list(sc_a.keys())}")

    # 2. FENSE
    if 'fense' in sc_a and 'fense' in sc_b:
        process_metric(model_name, "FENSE", sc_a['fense'].tolist(), sc_b['fense'].tolist(), standard_bins)
    
    # 3. SPIDEr-FL
    if 'spider_fl' in sc_a and 'spider_fl' in sc_b:
        process_metric(model_name, "SPIDEr-FL", sc_a['spider_fl'].tolist(), sc_b['spider_fl'].tolist(), spider_bins)
    
    # 4. CLAIR-A
    process_metric(model_name, "CLAIR-A", cl_a, cl_b, standard_bins)

# --- Main ---
if __name__ == "__main__":
    df_clotho, df_user, valid_keys = load_filtered_data()
    
    for name, path in CANDIDATE_FILES.items():
        run_evaluation(name, path, df_clotho, df_user)
        
    if excel_summary_data:
        logger.info(f"💾 Saving results to {OUTPUT_EXCEL_FILE}...")
        df_summary = pd.DataFrame(excel_summary_data)
        df_dist = pd.DataFrame(excel_distribution_data)
        summary_cols = ["Model", "Metric", "Mean_Scenario_A (Clotho)", "Mean_Scenario_B (User)", "Mean_Diff", "Correlation"]
        df_summary = df_summary[summary_cols]
        with pd.ExcelWriter(OUTPUT_EXCEL_FILE, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Summary_Stats', index=False)
            df_dist.to_excel(writer, sheet_name='Distributions', index=False)
        logger.info("✅ All Done!")
        print(df_summary)
    else:
        logger.error("❌ No data processed.")