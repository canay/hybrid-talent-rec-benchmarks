import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train_eval_talent_hybrid as hybrid


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def prepare_split(items, interactions, config):
    histories = hybrid.build_histories(interactions)
    item_ids, item_index, family_lookup, static_lookup = hybrid.build_indices(items)
    user_ids = sorted(histories)
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    train, val, test = hybrid.make_split(histories, mode="canonical", seed=config.canonical_seed)
    popularity = hybrid.build_popularity(train, item_ids, item_index)
    interaction_matrix = hybrid.build_interaction_matrix(train, user_ids, user_index, item_index)
    return {
        "histories": histories,
        "item_ids": item_ids,
        "item_index": item_index,
        "family_lookup": family_lookup,
        "static_lookup": static_lookup,
        "user_ids": user_ids,
        "user_index": user_index,
        "train": train,
        "val": val,
        "test": test,
        "popularity": popularity,
        "interaction_matrix": interaction_matrix,
    }


def benchmark_dataset(dataset_dir, repeats):
    config = hybrid.Config(dataset_dir=str(dataset_dir), output_path="unused")
    items, interactions, metadata = hybrid.load_dataset(config.dataset_dir)
    prepared = prepare_split(items, interactions, config)

    timing_rows = {
        "transition_cf": [],
        "gru4rec": [],
        "sasrec": [],
        "full_hybrid": [],
    }

    for run_idx in range(repeats):
        seed = config.canonical_seed + run_idx

        _, cf_seconds = timed(
            hybrid.transition_cf_scores,
            prepared["train"],
            prepared["user_ids"],
            prepared["item_ids"],
            prepared["item_index"],
            prepared["popularity"],
        )
        timing_rows["transition_cf"].append({"fit_select_seconds": cf_seconds, "test_eval_seconds": 0.0})

        gru_result, gru_seconds = timed(
            hybrid.fit_best_gru4rec,
            prepared["train"],
            prepared["user_ids"],
            prepared["val"],
            prepared["item_index"],
            config.top_k,
            config.gru_hidden_dims_grid,
            config,
            seed,
        )
        _, gru_eval_seconds = timed(
            hybrid.evaluate_scores,
            gru_result["scores"],
            prepared["test"],
            prepared["item_index"],
            config.top_k,
        )
        timing_rows["gru4rec"].append(
            {
                "fit_select_seconds": gru_seconds,
                "test_eval_seconds": gru_eval_seconds,
                "hidden_dim": gru_result["hidden_dim"],
            }
        )

        sasrec_result, sasrec_seconds = timed(
            hybrid.fit_best_sasrec,
            prepared["train"],
            prepared["user_ids"],
            prepared["val"],
            prepared["item_index"],
            config.top_k,
            config.sasrec_hidden_dims_grid,
            config,
            seed,
        )
        _, sasrec_eval_seconds = timed(
            hybrid.evaluate_scores,
            sasrec_result["scores"],
            prepared["test"],
            prepared["item_index"],
            config.top_k,
        )
        timing_rows["sasrec"].append(
            {
                "fit_select_seconds": sasrec_seconds,
                "test_eval_seconds": sasrec_eval_seconds,
                "hidden_dim": sasrec_result["hidden_dim"],
            }
        )

        def build_full_hybrid():
            feature_matrix, _ = hybrid.build_feature_matrix(
                prepared["train"],
                prepared["item_ids"],
                prepared["item_index"],
                prepared["static_lookup"],
            )
            global_weights = hybrid.compute_entropy_weights(feature_matrix)
            user_weights = hybrid.build_user_topsis_weights(
                prepared["train"], prepared["user_ids"], prepared["item_index"], feature_matrix
            )
            topsis_candidates = {
                alpha: hybrid.normalize_score_map(
                    hybrid.topsis_scores(feature_matrix, prepared["user_ids"], user_weights, global_weights, alpha)
                )
                for alpha in config.topsis_alphas
            }
            cf_map = hybrid.transition_cf_scores(
                prepared["train"],
                prepared["user_ids"],
                prepared["item_ids"],
                prepared["item_index"],
                prepared["popularity"],
            )
            rl_map = hybrid.build_family_transition_bandit(
                train=prepared["train"],
                user_ids=prepared["user_ids"],
                item_ids=prepared["item_ids"],
                family_lookup=prepared["family_lookup"],
                popularity=prepared["popularity"],
                learning_rate=config.rl_learning_rate,
                negative_reward=config.rl_negative_reward,
                negatives_per_positive=config.rl_negatives_per_positive,
                epochs=config.rl_epochs,
                transition_weight=config.rl_transition_weight,
                seed=seed,
            )
            return hybrid.select_best_full_hybrid(
                cf_map, rl_map, topsis_candidates, prepared["user_ids"], prepared["val"], prepared["item_index"], config
            )

        full_result, full_seconds = timed(build_full_hybrid)
        _, full_eval_seconds = timed(
            hybrid.evaluate_scores,
            full_result["scores"],
            prepared["test"],
            prepared["item_index"],
            config.top_k,
        )
        timing_rows["full_hybrid"].append(
            {
                "fit_select_seconds": full_seconds,
                "test_eval_seconds": full_eval_seconds,
                "weights": {
                    "cf_weight": full_result["cf_weight"],
                    "rl_weight": full_result["rl_weight"],
                    "topsis_weight": full_result["topsis_weight"],
                },
            }
        )

    summary = {}
    for model_name, runs in timing_rows.items():
        summary[model_name] = {
            "fit_select_seconds": {
                "mean": statistics.mean(run["fit_select_seconds"] for run in runs),
                "std": statistics.pstdev(run["fit_select_seconds"] for run in runs),
            },
            "test_eval_seconds": {
                "mean": statistics.mean(run["test_eval_seconds"] for run in runs),
                "std": statistics.pstdev(run["test_eval_seconds"] for run in runs),
            },
            "runs": runs,
        }

    return {
        "dataset_metadata": metadata,
        "config": config.__dict__,
        "runtime_summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark coarse runtime of selected recommendation models.")
    parser.add_argument(
        "--dataset-dir",
        action="append",
        help="Prepared benchmark directory. Can be passed multiple times.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-path",
        default=str(REPO_ROOT / "output" / "review" / "runtime_benchmark_results.json"),
    )
    args = parser.parse_args()

    dataset_dirs = args.dataset_dir or [
        str(REPO_ROOT / "data" / "prepared" / "jobhop_ict_benchmark_v1"),
        str(REPO_ROOT / "data" / "prepared" / "karrierewege_ict_benchmark_v1"),
    ]
    payload = {
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "repeats": args.repeats,
        },
        "datasets": {
            Path(dataset_dir).name: benchmark_dataset(Path(dataset_dir), repeats=args.repeats)
            for dataset_dir in dataset_dirs
        },
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote runtime benchmark results to {output_path.resolve()}")


if __name__ == "__main__":
    main()
