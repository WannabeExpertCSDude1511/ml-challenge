import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .io import read_tsv
from .blocking import generate_candidates
from .features import pair_features


def parse_truth(gt):
    out = {}
    for _, r in gt.iterrows():
        raw = r["matched_entity_ids"]
        out[r["source1_entity_id"]] = [x for x in raw.split(",") if x]
    return out


def create_model(model_type, y_train):
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError(
                "The 'xgboost' package is not installed. "
                "Install it via 'pip install xgboost' or run with '--model-type histgb'."
            )
        
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass

        pos_count = float(np.sum(y_train))
        neg_count = float(len(y_train) - pos_count)
        scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

        return XGBClassifier(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=6,
            reg_lambda=1.0,
            tree_method="hist",
            device=device,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
        )
    elif model_type == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Choose 'histgb' or 'xgboost'.")


def main():
    ap = argparse.ArgumentParser(description="Train business entity resolution classifier.")
    ap.add_argument("--data", default="dataset/train", help="Path to training data directory")
    ap.add_argument("--model", default="model.joblib", help="Output model joblib path")
    ap.add_argument("--model-type", choices=["histgb", "xgboost"], default="histgb", help="Classifier type: histgb or xgboost")
    args = ap.parse_args()

    d = Path(args.data)
    s1 = read_tsv(d / "train_source1.tsv")
    s2 = read_tsv(d / "train_source2.tsv")
    s3 = read_tsv(d / "train_source3.tsv")
    gt = parse_truth(read_tsv(d / "train_ground_truth.tsv"))
    target = pd.concat([s2, s3], ignore_index=True)

    pairs = generate_candidates(s1, target)
    target_by_id = target.set_index("entity_id")
    s1_by_id = s1.set_index("entity_id")

    X, y = [], []
    positives = set((a, b) for a, ids in gt.items() for b in ids)
    for a_id, b_id in pairs:
        x = pair_features(s1_by_id.loc[a_id], target_by_id.loc[b_id])
        X.append(x)
        y.append(int((a_id, b_id) in positives))

    X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = np.asarray(y)
    if y.sum() == 0:
        raise RuntimeError("No positive training pairs were found inside the candidate set.")

    model = create_model(args.model_type, y)
    model.fit(X, y)
    
    joblib.dump({"model": model, "features": list(X.columns), "model_type": args.model_type}, args.model)
    print(json.dumps({"pairs": len(X), "positives": int(y.sum()), "model": args.model, "model_type": args.model_type}, indent=2))


if __name__ == "__main__":
    main()
