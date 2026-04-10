import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train_eval_talent_hybrid as hybrid


@dataclass
class DQNConfig:
    hidden_dims_grid: tuple = (16, 32, 64)
    gamma: float = 0.4
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    target_sync_every: int = 5


class FamilyDQN(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states):
        return self.network(states)


def encode_state(prefix_items, family_lookup, family_index):
    family_count = len(family_index)
    counts = np.zeros(family_count, dtype=np.float32)
    for item_id in prefix_items:
        counts[family_index[family_lookup[item_id]]] += 1.0
    if counts.sum() > 0:
        counts /= counts.sum()
    last_family = np.zeros(family_count, dtype=np.float32)
    last_family[family_index[family_lookup[prefix_items[-1]]]] = 1.0
    return np.concatenate([last_family, counts]).astype(np.float32)


def build_dqn_experiences(train, family_lookup, family_index, negatives_per_positive, negative_reward, seed):
    rng = random.Random(seed)
    action_space = list(range(len(family_index)))
    experiences = []
    for sequence in train.values():
        if len(sequence) < 2:
            continue
        for step in range(len(sequence) - 1):
            prefix = sequence[: step + 1]
            next_prefix = sequence[: step + 2]
            state = encode_state(prefix, family_lookup, family_index)
            next_state = encode_state(next_prefix, family_lookup, family_index)
            action = family_index[family_lookup[sequence[step + 1]]]
            done = step + 1 == len(sequence) - 1
            experiences.append((state, action, 1.0, next_state, done))
            negatives = [candidate for candidate in action_space if candidate != action]
            for negative_action in rng.sample(negatives, min(negatives_per_positive, len(negatives))):
                experiences.append((state, negative_action, negative_reward, state, True))
    return experiences


def iterate_batches(experiences, batch_size, seed):
    rng = random.Random(seed)
    indices = list(range(len(experiences)))
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch = [experiences[idx] for idx in batch_indices]
        states = torch.tensor(np.stack([row[0] for row in batch]), dtype=torch.float32)
        actions = torch.tensor([row[1] for row in batch], dtype=torch.long)
        rewards = torch.tensor([row[2] for row in batch], dtype=torch.float32)
        next_states = torch.tensor(np.stack([row[3] for row in batch]), dtype=torch.float32)
        dones = torch.tensor([row[4] for row in batch], dtype=torch.bool)
        yield states, actions, rewards, next_states, dones


def build_dqn_score_map(model, train, user_ids, item_ids, family_lookup, family_index, popularity):
    model.eval()
    scores = {}
    with torch.no_grad():
        for user_id in user_ids:
            state = torch.tensor(encode_state(train[user_id], family_lookup, family_index), dtype=torch.float32).unsqueeze(0)
            family_scores = model(state).squeeze(0).cpu().numpy()
            family_scores = hybrid.normalize_array(family_scores)
            item_vector = np.zeros(len(item_ids), dtype=float)
            for idx, item_id in enumerate(item_ids):
                family_idx = family_index[family_lookup[item_id]]
                item_vector[idx] = 0.75 * family_scores[family_idx] + 0.25 * popularity[idx]
            scores[user_id] = hybrid.normalize_array(item_vector)
    return hybrid.normalize_score_map(scores)


def fit_best_dqn(train, user_ids, item_ids, family_lookup, val, item_index, top_k, base_config, dqn_config, seed):
    families = sorted({family_lookup[item_id] for item_id in item_ids})
    family_index = {family: idx for idx, family in enumerate(families)}
    experiences = build_dqn_experiences(
        train,
        family_lookup,
        family_index,
        base_config.rl_negatives_per_positive,
        base_config.rl_negative_reward,
        seed,
    )
    if not experiences:
        zero_scores = {user_id: np.zeros(len(item_ids), dtype=float) for user_id in user_ids}
        normalized = hybrid.normalize_score_map(zero_scores)
        return {
            "hidden_dim": None,
            "family_index": family_index,
            "scores": normalized,
            "metrics": hybrid.evaluate_scores(normalized, val, item_index, top_k),
        }

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    best = None
    state_dim = len(family_index) * 2
    action_dim = len(family_index)

    for hidden_dim in dqn_config.hidden_dims_grid:
        model = FamilyDQN(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim)
        target_model = FamilyDQN(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim)
        target_model.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=dqn_config.learning_rate, weight_decay=dqn_config.weight_decay)
        criterion = nn.SmoothL1Loss()

        for epoch in range(dqn_config.epochs):
            model.train()
            for batch_id, batch in enumerate(iterate_batches(experiences, dqn_config.batch_size, seed + hidden_dim + epoch)):
                states, actions, rewards, next_states, dones = batch
                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_max = target_model(next_states).max(dim=1).values
                    targets = rewards + dqn_config.gamma * next_max * (~dones).float()
                optimizer.zero_grad()
                loss = criterion(q_values, targets)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % dqn_config.target_sync_every == 0:
                target_model.load_state_dict(model.state_dict())

        scores = build_dqn_score_map(model, train, user_ids, item_ids, family_lookup, family_index, hybrid.build_popularity(train, item_ids, item_index))
        metrics = hybrid.evaluate_scores(scores, val, item_index, top_k)
        if best is None or metrics["summary"]["NDCG@K"] > best["metrics"]["summary"]["NDCG@K"]:
            best = {
                "hidden_dim": hidden_dim,
                "family_index": family_index,
                "scores": scores,
                "metrics": metrics,
            }
    return best


def run_split(seed, mode, items, histories, base_config, dqn_config):
    item_ids, item_index, family_lookup, static_lookup = hybrid.build_indices(items)
    train, val, test = hybrid.make_split(histories, mode=mode, seed=seed)
    user_ids = sorted(train)
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    popularity = hybrid.build_popularity(train, item_ids, item_index)
    interaction_matrix = hybrid.build_interaction_matrix(train, user_ids, user_index, item_index)
    feature_matrix, _ = hybrid.build_feature_matrix(train, item_ids, item_index, static_lookup)
    global_weights = hybrid.compute_entropy_weights(feature_matrix)
    user_weights = hybrid.build_user_topsis_weights(train, user_ids, item_index, feature_matrix)
    topsis_candidates = {
        alpha: hybrid.normalize_score_map(
            hybrid.topsis_scores(feature_matrix, user_ids, user_weights, global_weights, alpha)
        )
        for alpha in base_config.topsis_alphas
    }
    cf_map = hybrid.transition_cf_scores(train, user_ids, item_ids, item_index, popularity)
    rl_map = hybrid.build_family_transition_bandit(
        train=train,
        user_ids=user_ids,
        item_ids=item_ids,
        family_lookup=family_lookup,
        popularity=popularity,
        learning_rate=base_config.rl_learning_rate,
        negative_reward=base_config.rl_negative_reward,
        negatives_per_positive=base_config.rl_negatives_per_positive,
        epochs=base_config.rl_epochs,
        transition_weight=base_config.rl_transition_weight,
        seed=seed,
    )
    dqn_result = fit_best_dqn(train, user_ids, item_ids, family_lookup, val, item_index, base_config.top_k, base_config, dqn_config, seed)
    best_full = hybrid.select_best_full_hybrid(cf_map, rl_map, topsis_candidates, user_ids, val, item_index, base_config)
    best_full_dqn = hybrid.select_best_full_hybrid(cf_map, dqn_result["scores"], topsis_candidates, user_ids, val, item_index, base_config)
    model_scores = {
        "rl_bandit": rl_map,
        "dqn_family": dqn_result["scores"],
        "full_hybrid": best_full["scores"],
        "full_hybrid_dqn": best_full_dqn["scores"],
    }
    summary = {
        model_name: hybrid.evaluate_scores(score_map, test, item_index, base_config.top_k)["summary"]
        for model_name, score_map in model_scores.items()
    }
    tuned_params = {
        "dqn_hidden_dim": dqn_result["hidden_dim"],
        "full_hybrid": {
            "alpha": best_full["alpha"],
            "cf_weight": best_full["cf_weight"],
            "rl_weight": best_full["rl_weight"],
            "topsis_weight": best_full["topsis_weight"],
        },
        "full_hybrid_dqn": {
            "alpha": best_full_dqn["alpha"],
            "cf_weight": best_full_dqn["cf_weight"],
            "dqn_weight": best_full_dqn["rl_weight"],
            "topsis_weight": best_full_dqn["topsis_weight"],
        },
    }
    return {"seed": seed, "mode": mode, "metrics": summary, "tuned_params": tuned_params}


def summarize_metric_across_splits(split_results, metric_name):
    values = [result[metric_name] for result in split_results]
    return {"mean": statistics.mean(values), "std": statistics.pstdev(values), "values": values}


def aggregate_results(split_outputs):
    model_names = list(split_outputs[0]["metrics"].keys())
    aggregated = {}
    per_model_metric_table = {model_name: [] for model_name in model_names}
    for split_output in split_outputs:
        for model_name, metrics in split_output["metrics"].items():
            per_model_metric_table[model_name].append(metrics)
    for model_name, split_metrics in per_model_metric_table.items():
        aggregated[model_name] = {}
        for metric_name in ("HR@K", "NDCG@K", "MRR@K"):
            aggregated[model_name][metric_name] = summarize_metric_across_splits(split_metrics, metric_name)
    stat_tests = {
        "full_hybrid_vs_full_hybrid_dqn_ndcg": hybrid.wilcoxon_test(per_model_metric_table, "full_hybrid", "full_hybrid_dqn", "NDCG@K"),
        "dqn_family_vs_rl_bandit_ndcg": hybrid.wilcoxon_test(per_model_metric_table, "dqn_family", "rl_bandit", "NDCG@K"),
    }
    return aggregated, stat_tests


def print_report(title, aggregated_results, stat_tests):
    print("=" * 88)
    print(title)
    print("=" * 88)
    header = f"{'Model':<20} {'HR@K':>14} {'NDCG@K':>14} {'MRR@K':>14}"
    print(header)
    print("-" * len(header))
    for model_name, metrics in aggregated_results.items():
        print(
            f"{model_name:<20} "
            f"{metrics['HR@K']['mean']:.4f}+-{metrics['HR@K']['std']:.4f} "
            f"{metrics['NDCG@K']['mean']:.4f}+-{metrics['NDCG@K']['std']:.4f} "
            f"{metrics['MRR@K']['mean']:.4f}+-{metrics['MRR@K']['std']:.4f}"
        )
    print("-" * len(header))
    for test_name, result in stat_tests.items():
        print(f"{test_name}: statistic={result['statistic']:.4f}, p={result['pvalue']:.4f}, rbc={result['rank_biserial']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compare the tabular family bandit with a simple family-level DQN ablation.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=100)
    args = parser.parse_args()

    base_config = hybrid.Config(
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        repeats=args.repeats,
        seed_start=args.seed_start,
    )
    dqn_config = DQNConfig()
    items, interactions, metadata = hybrid.load_dataset(base_config.dataset_dir)
    histories = hybrid.build_histories(interactions)

    canonical_output = run_split(base_config.canonical_seed, "canonical", items, histories, base_config, dqn_config)
    repeated_outputs = [
        run_split(base_config.seed_start + offset, "repeated", items, histories, base_config, dqn_config)
        for offset in range(base_config.repeats)
    ]
    aggregated_results, stat_tests = aggregate_results(repeated_outputs)

    canonical_aggregated = {
        model_name: {
            metric_name: {"mean": metric_value, "std": 0.0, "values": [metric_value]}
            for metric_name, metric_value in metrics.items()
        }
        for model_name, metrics in canonical_output["metrics"].items()
    }
    print_report(f"{metadata['source_name']} CANONICAL", canonical_aggregated, {})
    print_report(f"{metadata['source_name']} REPEATED", aggregated_results, stat_tests)

    payload = {
        "dataset_metadata": metadata,
        "dqn_config": dqn_config.__dict__,
        "canonical": canonical_output,
        "repeated": {
            "splits": repeated_outputs,
            "aggregated_results": aggregated_results,
            "stat_tests": stat_tests,
        },
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote DQN ablation report to {output_path.resolve()}")


if __name__ == "__main__":
    main()
