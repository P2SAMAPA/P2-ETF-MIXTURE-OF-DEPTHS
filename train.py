import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from huggingface_hub import HfApi
import config
import data_manager as dm
from mixture_of_depths import mod_score

def normalize_scores(score_dict):
    scores = np.array(list(score_dict.values()))
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return {k: 0.0 for k in score_dict}
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-12:
        return {k: 0.5 for k in score_dict}
    norm = (scores - min_s) / (max_s - min_s)
    tickers = list(score_dict.keys())
    return {tickers[i]: float(norm[i]) for i in range(len(norm))}

def run_for_window(returns, macro_df, window_days):
    if len(returns) < window_days:
        return None
    ret_window = returns.iloc[-window_days:]
    if macro_df is None or macro_df.empty:
        return None
    macro_window = macro_df.loc[ret_window.index]
    if len(macro_window) < len(ret_window):
        return None
    raw_scores = {}
    for ticker in ret_window.columns:
        s = mod_score(
            ret_window[ticker],
            macro_window,
            hidden_size=config.HIDDEN_SIZE,
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            seq_len=config.SEQ_LEN,
            epochs=config.EPOCHS,
            lr=config.LEARNING_RATE,
            batch_size=config.BATCH_SIZE,
            threshold=config.DEPTH_THRESHOLD
        )
        if not np.isfinite(s):
            s = 0.0
        raw_scores[ticker] = float(s)
    norm_scores = normalize_scores(raw_scores)
    sorted_norm = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
    top_etfs = [{"ticker": t, "mod_score_norm": s, "raw_score": raw_scores[t]} for t, s in sorted_norm[:config.TOP_N]]
    return {
        "window": window_days,
        "top_etfs": top_etfs,
        "all_scores_raw": raw_scores,
        "all_scores_norm": norm_scores
    }

def main():
    print("Loading master data...")
    dm.load_master_data()
    macro_df = dm.get_macro_data()
    if macro_df is None:
        print("Error: No macro data found.")
        return
    results = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "windows": config.WINDOWS,
        "hidden_size": config.HIDDEN_SIZE,
        "num_heads": config.NUM_HEADS,
        "num_layers": config.NUM_LAYERS,
        "seq_len": config.SEQ_LEN,
        "epochs": config.EPOCHS,
        "depth_threshold": config.DEPTH_THRESHOLD,
        "universes": {}
    }
    for uni_name in config.UNIVERSES.keys():
        print(f"Processing {uni_name}...")
        returns = dm.get_universe_returns(uni_name)
        if returns.empty:
            print("  No data -> skipping")
            continue
        all_window_results = []
        for w in config.WINDOWS:
            print(f"  Window {w} days")
            out = run_for_window(returns, macro_df, w)
            if out:
                all_window_results.append(out)
            else:
                print(f"    Failed for window {w}")
        best_data = all_window_results[-1] if all_window_results else None
        results["universes"][uni_name] = {
            "best_window_data": best_data,
            "all_windows": all_window_results
        }
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"output/mod_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")
    api = HfApi(token=config.HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj=out_file,
            path_in_repo=os.path.basename(out_file),
            repo_id=config.OUTPUT_REPO,
            repo_type="dataset"
        )
        print(f"Uploaded to {config.OUTPUT_REPO}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
