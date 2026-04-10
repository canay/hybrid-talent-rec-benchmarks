import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train_eval_talent_hybrid as hybrid


def rank_of(scores, item_idx):
    order = np.argsort(-scores)
    return int(np.where(order == item_idx)[0][0]) + 1


def serialize_item(item_id, label_lookup, family_lookup):
    return {
        "item_id": item_id,
        "label": label_lookup[item_id],
        "family": family_lookup[item_id],
    }


def build_case(dataset_dir, user_id, output_path, top_n):
    config = hybrid.Config(dataset_dir=str(dataset_dir), output_path=str(output_path))
    items, interactions, metadata = hybrid.load_dataset(config.dataset_dir)
    histories = hybrid.build_histories(interactions)

    item_ids, item_index, family_lookup, static_lookup = hybrid.build_indices(items)
    label_lookup = dict(zip(items["item_id"], items["label"]))
    user_ids = sorted(histories)
    if user_id not in histories:
        raise KeyError(f"User {user_id!r} not found in {dataset_dir}")

    train, val, test = hybrid.make_split(histories, mode="canonical", seed=config.canonical_seed)
    popularity = hybrid.build_popularity(train, item_ids, item_index)
    feature_matrix, feature_names = hybrid.build_feature_matrix(train, item_ids, item_index, static_lookup)
    global_weights = hybrid.compute_entropy_weights(feature_matrix)
    user_weights = hybrid.build_user_topsis_weights(train, user_ids, item_index, feature_matrix)
    topsis_candidates = {
        alpha: hybrid.normalize_score_map(
            hybrid.topsis_scores(feature_matrix, user_ids, user_weights, global_weights, alpha)
        )
        for alpha in config.topsis_alphas
    }

    markov_map = hybrid.build_item_markov_scores(train, user_ids, item_ids, item_index, popularity)
    cf_map = hybrid.transition_cf_scores(train, user_ids, item_ids, item_index, popularity)
    rl_map = hybrid.build_family_transition_bandit(
        train=train,
        user_ids=user_ids,
        item_ids=item_ids,
        family_lookup=family_lookup,
        popularity=popularity,
        learning_rate=config.rl_learning_rate,
        negative_reward=config.rl_negative_reward,
        negatives_per_positive=config.rl_negatives_per_positive,
        epochs=config.rl_epochs,
        transition_weight=config.rl_transition_weight,
        seed=config.canonical_seed,
    )

    best_topsis = hybrid.select_best_topsis(
        user_ids, user_weights, global_weights, feature_matrix, val, item_index, config
    )
    best_cf_topsis = hybrid.select_best_cf_topsis(
        cf_map, topsis_candidates, user_ids, val, item_index, config
    )
    best_full = hybrid.select_best_full_hybrid(
        cf_map, rl_map, topsis_candidates, user_ids, val, item_index, config
    )

    target_item_id = test[user_id]
    target_item_idx = item_index[target_item_id]
    full_scores = best_full["scores"][user_id]
    cf_scores = cf_map[user_id]
    rl_scores = rl_map[user_id]
    topsis_scores = topsis_candidates[best_full["alpha"]][user_id]
    top_indices = np.argsort(-full_scores)[:top_n]
    weighted_cf = best_full["cf_weight"] * cf_scores
    weighted_rl = best_full["rl_weight"] * rl_scores
    weighted_topsis = best_full["topsis_weight"] * topsis_scores

    case = {
        "dataset": metadata["dataset_name"],
        "source_name": metadata["source_name"],
        "mode": "canonical",
        "seed": config.canonical_seed,
        "user_id": user_id,
        "history": [serialize_item(item_id, label_lookup, family_lookup) for item_id in train[user_id]],
        "validation_item": serialize_item(val[user_id], label_lookup, family_lookup),
        "test_item": serialize_item(target_item_id, label_lookup, family_lookup),
        "full_hybrid": {
            "alpha": float(best_full["alpha"]),
            "cf_weight": float(best_full["cf_weight"]),
            "rl_weight": float(best_full["rl_weight"]),
            "topsis_weight": float(best_full["topsis_weight"]),
        },
        "criteria_weights": [
            {"criterion": name, "weight": float(weight)}
            for name, weight in sorted(
                zip(feature_names, user_weights[user_id]), key=lambda pair: -pair[1]
            )
        ],
        "target_ranks": {
            "full_hybrid": rank_of(full_scores, target_item_idx),
            "hybrid_cf_topsis": rank_of(best_cf_topsis["scores"][user_id], target_item_idx),
            "item_markov": rank_of(markov_map[user_id], target_item_idx),
            "transition_cf": rank_of(cf_scores, target_item_idx),
            "rl_bandit": rank_of(rl_scores, target_item_idx),
            "topsis": rank_of(topsis_scores, target_item_idx),
        },
        "top_candidates": [],
    }

    for rank, candidate_idx in enumerate(top_indices, start=1):
        candidate_item_id = item_ids[candidate_idx]
        case["top_candidates"].append(
            {
                "rank": rank,
                "item": serialize_item(candidate_item_id, label_lookup, family_lookup),
                "is_test_item": candidate_item_id == target_item_id,
                "scores": {
                    "cf": float(cf_scores[candidate_idx]),
                    "rl": float(rl_scores[candidate_idx]),
                    "topsis": float(topsis_scores[candidate_idx]),
                    "full_hybrid": float(full_scores[candidate_idx]),
                },
                "weighted_components": {
                    "cf": float(weighted_cf[candidate_idx]),
                    "rl": float(weighted_rl[candidate_idx]),
                    "topsis": float(weighted_topsis[candidate_idx]),
                },
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(case, indent=2), encoding="utf-8")
    return case


def main():
    parser = argparse.ArgumentParser(description="Generate a reproducible interpretability case study artifact.")
    parser.add_argument(
        "--dataset-dir",
        default=str(REPO_ROOT / "data" / "prepared" / "jobhop_ict_benchmark_v1"),
    )
    parser.add_argument(
        "--user-id",
        default="57220",
        help="Anonymized user identifier from the prepared benchmark.",
    )
    parser.add_argument(
        "--output-path",
        default=str(REPO_ROOT / "output" / "review" / "jobhop_interpretability_case_57220.json"),
    )
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    case = build_case(
        dataset_dir=Path(args.dataset_dir),
        user_id=str(args.user_id),
        output_path=Path(args.output_path),
        top_n=args.top_n,
    )
    print(
        "Wrote interpretability case for user "
        f"{case['user_id']} to {Path(args.output_path).resolve()}"
    )


if __name__ == "__main__":
    main()
