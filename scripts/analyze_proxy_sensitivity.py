import argparse
import json
import sys
from pathlib import Path

import statistics


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train_eval_talent_hybrid as hybrid


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "values": values,
    }


def run_full_hybrid_for_features(items, histories, config, feature_names, seed, mode):
    item_ids, item_index, family_lookup, static_lookup = hybrid.build_indices(items)
    user_ids = sorted(histories)
    train, val, test = hybrid.make_split(histories, mode=mode, seed=seed)
    popularity = hybrid.build_popularity(train, item_ids, item_index)
    feature_matrix, selected_feature_names = hybrid.build_feature_matrix(
        train, item_ids, item_index, static_lookup, selected_features=feature_names
    )
    global_weights = hybrid.compute_entropy_weights(feature_matrix)
    user_weights = hybrid.build_user_topsis_weights(train, user_ids, item_index, feature_matrix)
    topsis_candidates = {
        alpha: hybrid.normalize_score_map(
            hybrid.topsis_scores(feature_matrix, user_ids, user_weights, global_weights, alpha)
        )
        for alpha in config.topsis_alphas
    }
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
        seed=seed,
    )
    best_full = hybrid.select_best_full_hybrid(
        cf_map, rl_map, topsis_candidates, user_ids, val, item_index, config
    )
    metrics = hybrid.evaluate_scores(best_full["scores"], test, item_index, config.top_k)["summary"]
    return {
        "metrics": metrics,
        "selected_features": selected_feature_names,
        "weights": {
            "cf_weight": best_full["cf_weight"],
            "rl_weight": best_full["rl_weight"],
            "topsis_weight": best_full["topsis_weight"],
            "alpha": best_full["alpha"],
        },
    }


def analyze_dataset(dataset_dir, repeats, seed_start):
    config = hybrid.Config(dataset_dir=str(dataset_dir), output_path="unused", repeats=repeats, seed_start=seed_start)
    items, interactions, metadata = hybrid.load_dataset(config.dataset_dir)
    histories = hybrid.build_histories(interactions)
    _, _, _, static_lookup = hybrid.build_indices(items)
    all_feature_names = list(
        hybrid.build_feature_matrix(
            hybrid.make_split(histories, mode="canonical", seed=config.canonical_seed)[0],
            items["item_id"].tolist(),
            {item_id: idx for idx, item_id in enumerate(items["item_id"].tolist())},
            static_lookup,
        )[1]
    )

    feature_sets = [("all_features", all_feature_names)]
    feature_sets.extend(
        [
            (f"drop_{feature_name}", [name for name in all_feature_names if name != feature_name])
            for feature_name in all_feature_names
        ]
    )

    repeated = {}
    for label, selected_features in feature_sets:
        split_outputs = []
        for offset in range(repeats):
            seed = seed_start + offset
            split_outputs.append(
                run_full_hybrid_for_features(
                    items,
                    histories,
                    config,
                    feature_names=selected_features,
                    seed=seed,
                    mode="repeated",
                )
            )
        repeated[label] = {
            "selected_features": selected_features,
            "aggregated_metrics": {
                metric_name: summarize([split["metrics"][metric_name] for split in split_outputs])
                for metric_name in ["HR@K", "NDCG@K", "MRR@K", "Precision@K"]
            },
            "mean_weights": {
                key: statistics.mean([split["weights"][key] for split in split_outputs])
                for key in ["cf_weight", "rl_weight", "topsis_weight", "alpha"]
            },
        }

    baseline_ndcg = repeated["all_features"]["aggregated_metrics"]["NDCG@K"]["mean"]
    proxy_drop_summary = []
    for label, result in repeated.items():
        if label == "all_features":
            continue
        removed = label.replace("drop_", "", 1)
        ndcg_mean = result["aggregated_metrics"]["NDCG@K"]["mean"]
        proxy_drop_summary.append(
            {
                "removed_proxy": removed,
                "ndcg_mean": ndcg_mean,
                "ndcg_std": result["aggregated_metrics"]["NDCG@K"]["std"],
                "delta_vs_all": ndcg_mean - baseline_ndcg,
            }
        )
    proxy_drop_summary.sort(key=lambda row: row["delta_vs_all"])

    return {
        "dataset_metadata": metadata,
        "all_feature_names": all_feature_names,
        "repeated_results": repeated,
        "proxy_drop_summary": proxy_drop_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Run leave-one-proxy-out sensitivity analysis for the TOPSIS branch.")
    parser.add_argument(
        "--dataset-dir",
        action="append",
        help="Prepared benchmark directory. Can be passed multiple times.",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument(
        "--output-path",
        default=str(REPO_ROOT / "output" / "review" / "proxy_sensitivity_results.json"),
    )
    args = parser.parse_args()

    dataset_dirs = args.dataset_dir or [
        str(REPO_ROOT / "data" / "prepared" / "jobhop_ict_benchmark_v1"),
        str(REPO_ROOT / "data" / "prepared" / "karrierewege_ict_benchmark_v1"),
    ]
    payload = {
        Path(dataset_dir).name: analyze_dataset(Path(dataset_dir), repeats=args.repeats, seed_start=args.seed_start)
        for dataset_dir in dataset_dirs
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote proxy sensitivity results to {output_path.resolve()}")


if __name__ == "__main__":
    main()
