import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from scipy.stats import rankdata
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Config:
    dataset_dir: str
    output_path: str
    top_k: int = 5
    repeats: int = 10
    seed_start: int = 100
    topsis_alphas: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    rl_weights: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    cf_weights: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    nmf_components_grid: tuple = (6, 10, 14)
    svd_components_grid: tuple = (6, 10, 14)
    gru_hidden_dims_grid: tuple = (16, 32, 64)
    sasrec_hidden_dims_grid: tuple = (16, 32, 64)
    rl_learning_rate: float = 0.2
    rl_negative_reward: float = -0.2
    rl_negatives_per_positive: int = 2
    rl_epochs: int = 30
    rl_transition_weight: float = 0.7
    canonical_seed: int = 20260331
    gru_epochs: int = 25
    gru_learning_rate: float = 1e-3
    gru_weight_decay: float = 1e-5
    gru_batch_size: int = 64
    sasrec_num_heads: int = 2
    sasrec_num_layers: int = 1
    sasrec_dropout: float = 0.1
    sasrec_epochs: int = 25
    sasrec_learning_rate: float = 1e-3
    sasrec_weight_decay: float = 1e-5
    sasrec_batch_size: int = 64


class GRU4Rec(nn.Module):
    def __init__(self, num_items, embedding_dim, hidden_dim, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_items)

    def forward(self, sequences):
        embedded = self.embedding(sequences)
        _, hidden = self.gru(embedded)
        logits = self.output(hidden[-1])
        return logits


class SASRec(nn.Module):
    def __init__(self, num_items, embedding_dim, padding_idx, max_len, num_heads, num_layers, dropout):
        super().__init__()
        self.padding_idx = padding_idx
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=padding_idx)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(embedding_dim, num_items)

    def forward(self, sequences):
        batch_size, seq_len = sequences.shape
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0).expand(batch_size, seq_len)
        padding_mask = sequences.eq(self.padding_idx)
        embedded = self.item_embedding(sequences) + self.position_embedding(positions)
        embedded = self.dropout(embedded)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=sequences.device, dtype=torch.bool),
            diagonal=1,
        )
        encoded = self.encoder(embedded, mask=causal_mask, src_key_padding_mask=padding_mask)
        valid_lengths = (~padding_mask).sum(dim=1).clamp(min=1) - 1
        last_hidden = encoded[torch.arange(batch_size, device=sequences.device), valid_lengths]
        return self.output(last_hidden)


def normalize_array(values, invert=False):
    arr = np.asarray(values, dtype=float)
    if invert:
        arr = arr.max() - arr
    lo = arr.min()
    hi = arr.max()
    if hi - lo < 1e-12:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - lo) / (hi - lo)


def normalize_score_map(score_map):
    return {user_id: normalize_array(scores) for user_id, scores in score_map.items()}


def load_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    items = pd.read_csv(dataset_dir / "items.csv")
    interactions = pd.read_csv(dataset_dir / "interactions.csv")
    with open(dataset_dir / "metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return items, interactions, metadata


def build_histories(interactions):
    histories = {}
    for user_id, group in interactions.groupby("user_id"):
        group = group.sort_values("sequence_pos")
        histories[str(user_id)] = group["item_id"].tolist()
    return histories


def build_indices(items):
    item_ids = items["item_id"].tolist()
    item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    family_lookup = dict(zip(items["item_id"], items["family"]))
    static_lookup = items.set_index("item_id").to_dict(orient="index")
    return item_ids, item_index, family_lookup, static_lookup


def make_split(histories, mode, seed):
    rng = random.Random(seed)
    train = {}
    val = {}
    test = {}
    for user_id, sequence in histories.items():
        if len(sequence) < 3:
            continue
        if mode == "canonical":
            test_index = len(sequence) - 1
        else:
            test_index = 2 if len(sequence) == 3 else rng.randint(2, len(sequence) - 1)
        val_index = test_index - 1
        train[user_id] = sequence[:val_index]
        val[user_id] = sequence[val_index]
        test[user_id] = sequence[test_index]
    return train, val, test


def build_interaction_matrix(train, user_ids, user_index, item_index):
    matrix = np.zeros((len(user_ids), len(item_index)), dtype=float)
    for user_id in user_ids:
        for item_id in set(train[user_id]):
            matrix[user_index[user_id], item_index[item_id]] = 1.0
    return matrix


def build_popularity(train, item_ids, item_index):
    counts = np.zeros(len(item_ids), dtype=float)
    for sequence in train.values():
        for item_id in sequence:
            counts[item_index[item_id]] += 1.0
    return normalize_array(counts)


def build_repeat_last_scores(train, user_ids, item_ids, item_index, popularity):
    scores = {}
    for user_id in user_ids:
        vector = 0.05 * popularity.copy()
        vector[item_index[train[user_id][-1]]] = 1.0
        scores[user_id] = vector
    return scores


def build_item_markov_scores(train, user_ids, item_ids, item_index, popularity):
    transition = np.zeros((len(item_ids), len(item_ids)), dtype=float)
    for sequence in train.values():
        for current_item, next_item in zip(sequence[:-1], sequence[1:]):
            transition[item_index[current_item], item_index[next_item]] += 1.0

    row_sums = transition.sum(axis=1, keepdims=True)
    probs = np.divide(transition, row_sums, out=np.zeros_like(transition), where=row_sums > 0)

    scores = {}
    for user_id in user_ids:
        last_item = train[user_id][-1]
        vector = probs[item_index[last_item]].copy()
        if vector.sum() < 1e-12:
            vector = popularity.copy()
        scores[user_id] = normalize_array(0.9 * vector + 0.1 * popularity)
    return scores


def transition_cf_scores(train, user_ids, item_ids, item_index, popularity):
    transition = np.zeros((len(item_ids), len(item_ids)), dtype=float)
    for sequence in train.values():
        if len(sequence) < 2:
            continue
        weights = np.arange(1, len(sequence), dtype=float)
        weights = weights / weights.sum()
        for (current_item, next_item), weight in zip(zip(sequence[:-1], sequence[1:]), weights):
            transition[item_index[current_item], item_index[next_item]] += weight

    similarity = cosine_similarity(transition + 1e-9)
    np.fill_diagonal(similarity, 0.0)

    scores = {}
    for user_id in user_ids:
        history = train[user_id]
        recency = np.arange(1, len(history) + 1, dtype=float)
        recency = recency / recency.sum()
        last_item = history[-1]
        markov_component = transition[item_index[last_item]]
        markov_component = normalize_array(markov_component) if markov_component.sum() > 0 else popularity
        transition_component = np.zeros(len(item_ids), dtype=float)
        similarity_component = np.zeros(len(item_ids), dtype=float)
        for weight, item_id in zip(recency, history):
            row = transition[item_index[item_id]]
            row = normalize_array(row) if row.sum() > 0 else popularity
            transition_component += weight * row
            similarity_component += weight * similarity[item_index[item_id]]
        history_component = normalize_array(
            0.7 * normalize_array(transition_component)
            + 0.3 * normalize_array(similarity_component)
            + 0.1 * popularity
        )
        scores[user_id] = normalize_array(0.85 * markov_component + 0.15 * history_component)
    return scores


def fit_best_nmf(interaction_matrix, user_ids, user_index, val, item_index, top_k, components_grid, seed):
    best = None
    for n_components in components_grid:
        n_components = min(n_components, interaction_matrix.shape[1] - 1)
        if n_components < 2:
            continue
        model = NMF(n_components=n_components, init="nndsvda", random_state=seed, max_iter=600)
        user_factors = model.fit_transform(interaction_matrix)
        predictions = user_factors @ model.components_
        scores = {user_id: predictions[user_index[user_id]].copy() for user_id in user_ids}
        scores = normalize_score_map(scores)
        metrics = evaluate_scores(scores, val, item_index, top_k)
        if best is None or metrics["summary"]["NDCG@K"] > best["metrics"]["summary"]["NDCG@K"]:
            best = {"n_components": n_components, "scores": scores, "metrics": metrics}
    return best


def fit_best_svd(interaction_matrix, user_ids, user_index, val, item_index, top_k, components_grid, seed):
    best = None
    for n_components in components_grid:
        n_components = min(n_components, interaction_matrix.shape[1] - 1)
        if n_components < 2:
            continue
        model = TruncatedSVD(n_components=n_components, random_state=seed)
        user_factors = model.fit_transform(interaction_matrix)
        predictions = user_factors @ model.components_
        scores = {user_id: predictions[user_index[user_id]].copy() for user_id in user_ids}
        scores = normalize_score_map(scores)
        metrics = evaluate_scores(scores, val, item_index, top_k)
        if best is None or metrics["summary"]["NDCG@K"] > best["metrics"]["summary"]["NDCG@K"]:
            best = {"n_components": n_components, "scores": scores, "metrics": metrics}
    return best


def build_sequential_training_examples(train, item_index):
    examples = []
    for sequence in train.values():
        encoded = [item_index[item_id] for item_id in sequence]
        for end_idx in range(1, len(encoded)):
            examples.append((encoded[:end_idx], encoded[end_idx]))
    return examples


def iterate_minibatches(examples, batch_size, padding_idx, rng):
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch = [examples[idx] for idx in batch_indices]
        max_len = max(len(seq) for seq, _ in batch)
        sequences = np.full((len(batch), max_len), padding_idx, dtype=np.int64)
        targets = np.zeros(len(batch), dtype=np.int64)
        for row_idx, (seq, target) in enumerate(batch):
            sequences[row_idx, : len(seq)] = seq
            targets[row_idx] = target
        yield torch.from_numpy(sequences), torch.from_numpy(targets)


def build_gru_score_map(model, train, user_ids, item_index, item_count, padding_idx, device):
    model.eval()
    scores = {}
    with torch.no_grad():
        for user_id in user_ids:
            encoded = [item_index[item_id] for item_id in train[user_id]]
            if not encoded:
                scores[user_id] = np.zeros(item_count, dtype=float)
                continue
            sequence = torch.full((1, len(encoded)), padding_idx, dtype=torch.long, device=device)
            sequence[0, : len(encoded)] = torch.tensor(encoded, dtype=torch.long, device=device)
            logits = model(sequence).squeeze(0).cpu().numpy()
            scores[user_id] = logits
    return normalize_score_map(scores)


def fit_best_gru4rec(train, user_ids, val, item_index, top_k, hidden_dims_grid, config, seed):
    examples = build_sequential_training_examples(train, item_index)
    padding_idx = len(item_index)
    item_count = len(item_index)
    if not examples:
        zero_scores = {user_id: np.zeros(item_count, dtype=float) for user_id in user_ids}
        return {"hidden_dim": None, "scores": normalize_score_map(zero_scores), "metrics": evaluate_scores(normalize_score_map(zero_scores), val, item_index, top_k)}

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")
    best = None

    for hidden_dim in hidden_dims_grid:
        model = GRU4Rec(
            num_items=item_count,
            embedding_dim=hidden_dim,
            hidden_dim=hidden_dim,
            padding_idx=padding_idx,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.gru_learning_rate,
            weight_decay=config.gru_weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        rng = random.Random(seed + hidden_dim)

        model.train()
        for _ in range(config.gru_epochs):
            for batch_sequences, batch_targets in iterate_minibatches(
                examples, config.gru_batch_size, padding_idx, rng
            ):
                batch_sequences = batch_sequences.to(device)
                batch_targets = batch_targets.to(device)
                optimizer.zero_grad()
                logits = model(batch_sequences)
                loss = criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()

        scores = build_gru_score_map(model, train, user_ids, item_index, item_count, padding_idx, device)
        metrics = evaluate_scores(scores, val, item_index, top_k)
        if best is None or metrics["summary"]["NDCG@K"] > best["metrics"]["summary"]["NDCG@K"]:
            best = {"hidden_dim": hidden_dim, "scores": scores, "metrics": metrics}
    return best


def build_sasrec_score_map(model, train, user_ids, item_index, item_count, padding_idx, device):
    model.eval()
    scores = {}
    with torch.no_grad():
        for user_id in user_ids:
            encoded = [item_index[item_id] for item_id in train[user_id]]
            if not encoded:
                scores[user_id] = np.zeros(item_count, dtype=float)
                continue
            sequence = torch.full((1, len(encoded)), padding_idx, dtype=torch.long, device=device)
            sequence[0, : len(encoded)] = torch.tensor(encoded, dtype=torch.long, device=device)
            logits = model(sequence).squeeze(0).cpu().numpy()
            scores[user_id] = logits
    return normalize_score_map(scores)


def fit_best_sasrec(train, user_ids, val, item_index, top_k, hidden_dims_grid, config, seed):
    examples = build_sequential_training_examples(train, item_index)
    padding_idx = len(item_index)
    item_count = len(item_index)
    if not examples:
        zero_scores = {user_id: np.zeros(item_count, dtype=float) for user_id in user_ids}
        normalized = normalize_score_map(zero_scores)
        return {
            "hidden_dim": None,
            "scores": normalized,
            "metrics": evaluate_scores(normalized, val, item_index, top_k),
        }

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")
    max_len = max(max(len(sequence) for sequence, _ in examples), max(len(sequence) for sequence in train.values()))
    best = None

    for hidden_dim in hidden_dims_grid:
        if hidden_dim % config.sasrec_num_heads != 0:
            continue
        model = SASRec(
            num_items=item_count,
            embedding_dim=hidden_dim,
            padding_idx=padding_idx,
            max_len=max_len,
            num_heads=config.sasrec_num_heads,
            num_layers=config.sasrec_num_layers,
            dropout=config.sasrec_dropout,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.sasrec_learning_rate,
            weight_decay=config.sasrec_weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        rng = random.Random(seed + 1000 + hidden_dim)

        model.train()
        for _ in range(config.sasrec_epochs):
            for batch_sequences, batch_targets in iterate_minibatches(
                examples, config.sasrec_batch_size, padding_idx, rng
            ):
                batch_sequences = batch_sequences.to(device)
                batch_targets = batch_targets.to(device)
                optimizer.zero_grad()
                logits = model(batch_sequences)
                loss = criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()

        scores = build_sasrec_score_map(model, train, user_ids, item_index, item_count, padding_idx, device)
        metrics = evaluate_scores(scores, val, item_index, top_k)
        if best is None or metrics["summary"]["NDCG@K"] > best["metrics"]["summary"]["NDCG@K"]:
            best = {"hidden_dim": hidden_dim, "scores": scores, "metrics": metrics}
    return best


def build_feature_matrix(train, item_ids, item_index, static_lookup, selected_features=None):
    interaction_count = np.zeros(len(item_ids), dtype=float)
    transition_out = defaultdict(set)
    for sequence in train.values():
        for item_id in sequence:
            interaction_count[item_index[item_id]] += 1.0
        for current_item, next_item in zip(sequence[:-1], sequence[1:]):
            transition_out[current_item].add(next_item)

    prevalence = normalize_array(interaction_count)
    skill_breadth = normalize_array([static_lookup[item_id]["skill_count"] for item_id in item_ids])
    digital_density = normalize_array([static_lookup[item_id]["digital_skill_density"] for item_id in item_ids])
    innovation = normalize_array([static_lookup[item_id]["innovation_score"] for item_id in item_ids])
    role_level = normalize_array([static_lookup[item_id]["role_level_score"] for item_id in item_ids])
    mobility = normalize_array([len(transition_out[item_id]) for item_id in item_ids])

    feature_lookup = {
        "market_prevalence": prevalence,
        "skill_breadth": skill_breadth,
        "digital_skill_density": digital_density,
        "innovation_intensity": innovation,
        "role_level": role_level,
        "transition_mobility": mobility,
    }
    feature_names = list(feature_lookup.keys()) if selected_features is None else list(selected_features)
    feature_matrix = np.vstack([feature_lookup[name] for name in feature_names]).T
    return feature_matrix, feature_names


def compute_entropy_weights(feature_matrix):
    safe = np.maximum(feature_matrix, 1e-12)
    proportions = safe / safe.sum(axis=0, keepdims=True)
    entropy = -(proportions * np.log(proportions)).sum(axis=0) / (math.log(feature_matrix.shape[0]) + 1e-12)
    diversity = 1.0 - entropy
    return diversity / diversity.sum()


def build_user_topsis_weights(train, user_ids, item_index, feature_matrix):
    global_mean = feature_matrix.mean(axis=0)
    global_std = feature_matrix.std(axis=0) + 1e-9
    user_weights = {}
    for user_id in user_ids:
        train_items = train[user_id]
        indices = [item_index[item_id] for item_id in train_items]
        if not indices:
            user_weights[user_id] = np.full(feature_matrix.shape[1], 1.0 / feature_matrix.shape[1])
            continue
        recency = np.arange(1, len(indices) + 1, dtype=float)
        recency = recency / recency.sum()
        weighted_mean = (feature_matrix[indices] * recency[:, None]).sum(axis=0)
        logits = np.exp((weighted_mean - global_mean) / global_std)
        user_weights[user_id] = logits / logits.sum()
    return user_weights


def topsis_scores(feature_matrix, user_ids, user_weights, global_weights, alpha):
    normalized = feature_matrix / np.sqrt((feature_matrix ** 2).sum(axis=0, keepdims=True) + 1e-12)
    scores = {}
    for user_id in user_ids:
        weights = alpha * user_weights[user_id] + (1.0 - alpha) * global_weights
        weighted = normalized * weights
        ideal_best = weighted.max(axis=0)
        ideal_worst = weighted.min(axis=0)
        distance_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
        distance_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
        scores[user_id] = distance_worst / (distance_best + distance_worst + 1e-12)
    return scores


def build_family_transition_bandit(
    train,
    user_ids,
    item_ids,
    family_lookup,
    popularity,
    learning_rate,
    negative_reward,
    negatives_per_positive,
    epochs,
    transition_weight,
    seed,
):
    rng = random.Random(seed)
    families = sorted({family_lookup[item_id] for item_id in item_ids})
    family_index = {family: idx for idx, family in enumerate(families)}
    q_table = np.zeros((len(families), len(families)), dtype=float)

    for _ in range(epochs):
        for sequence in train.values():
            if len(sequence) < 2:
                continue
            for current_item, next_item in zip(sequence[:-1], sequence[1:]):
                state = family_index[family_lookup[current_item]]
                action = family_index[family_lookup[next_item]]
                q_table[state, action] += learning_rate * (1.0 - q_table[state, action])
                negatives = [idx for idx in range(len(families)) if idx != action]
                for neg in rng.sample(negatives, min(negatives_per_positive, len(negatives))):
                    q_table[state, neg] += learning_rate * (negative_reward - q_table[state, neg])

    row_mins = q_table.min(axis=1, keepdims=True)
    row_maxs = q_table.max(axis=1, keepdims=True)
    q_table = (q_table - row_mins) / (row_maxs - row_mins + 1e-12)

    scores = {}
    for user_id in user_ids:
        history = train[user_id]
        last_family = family_lookup[history[-1]]
        state_scores = q_table[family_index[last_family]]
        recency_weights = np.arange(1, len(history) + 1, dtype=float)
        recency_weights = recency_weights / recency_weights.sum()
        family_bias = np.zeros(len(families), dtype=float)
        for weight, item_id in zip(recency_weights, history):
            family_bias[family_index[family_lookup[item_id]]] += weight
        family_vector = transition_weight * state_scores + (1.0 - transition_weight) * normalize_array(family_bias)
        item_vector = np.zeros(len(item_ids), dtype=float)
        for idx, item_id in enumerate(item_ids):
            item_vector[idx] = 0.75 * family_vector[family_index[family_lookup[item_id]]] + 0.25 * popularity[idx]
        scores[user_id] = normalize_array(item_vector)
    return scores


def blend_scores(score_map_a, weight_a, score_map_b, weight_b, user_ids):
    return {user_id: weight_a * score_map_a[user_id] + weight_b * score_map_b[user_id] for user_id in user_ids}


def blend_three_scores(score_map_a, weight_a, score_map_b, weight_b, score_map_c, weight_c, user_ids):
    return {
        user_id: (
            weight_a * score_map_a[user_id]
            + weight_b * score_map_b[user_id]
            + weight_c * score_map_c[user_id]
        )
        for user_id in user_ids
    }


def evaluate_scores(score_map, holdout, item_index, top_k):
    hr_values = []
    ndcg_values = []
    mrr_values = []
    precision_values = []
    for user_id, scores in score_map.items():
        ranked = np.argsort(-scores)[:top_k]
        ground_truth = item_index[holdout[user_id]]
        hr = 1.0 if ground_truth in ranked else 0.0
        precision = 1.0 / top_k if hr else 0.0
        rr = 0.0
        dcg = 0.0
        for rank, item_idx in enumerate(ranked, start=1):
            if item_idx == ground_truth:
                rr = 1.0 / rank
                dcg = 1.0 / math.log2(rank + 1)
                break
        hr_values.append(hr)
        precision_values.append(precision)
        ndcg_values.append(dcg)
        mrr_values.append(rr)
    summary = {
        "HR@K": float(np.mean(hr_values)),
        "NDCG@K": float(np.mean(ndcg_values)),
        "MRR@K": float(np.mean(mrr_values)),
        "Precision@K": float(np.mean(precision_values)),
    }
    per_user = {
        "HR@K": hr_values,
        "NDCG@K": ndcg_values,
        "MRR@K": mrr_values,
        "Precision@K": precision_values,
    }
    return {"summary": summary, "per_user": per_user}


def select_best_topsis(user_ids, user_weights, global_weights, feature_matrix, val, item_index, config):
    best = None
    for alpha in config.topsis_alphas:
        scores = normalize_score_map(topsis_scores(feature_matrix, user_ids, user_weights, global_weights, alpha))
        metrics = evaluate_scores(scores, val, item_index, config.top_k)
        if best is None or metrics["summary"]["NDCG@K"] > best["metrics"]["summary"]["NDCG@K"]:
            best = {"alpha": alpha, "scores": scores, "metrics": metrics}
    return best


def select_best_cf_topsis(cf_scores_map, topsis_candidates, user_ids, val, item_index, config):
    best = None
    for alpha, topsis_map in topsis_candidates.items():
        for cf_weight in config.cf_weights:
            topsis_weight = 1.0 - cf_weight
            scores = normalize_score_map(blend_scores(cf_scores_map, cf_weight, topsis_map, topsis_weight, user_ids))
            metrics = evaluate_scores(scores, val, item_index, config.top_k)
            if best is None or metrics["summary"]["NDCG@K"] > best["metrics"]["summary"]["NDCG@K"]:
                best = {
                    "alpha": alpha,
                    "cf_weight": cf_weight,
                    "topsis_weight": topsis_weight,
                    "scores": scores,
                    "metrics": metrics,
                }
    return best


def select_best_rl_topsis(rl_scores_map, topsis_candidates, user_ids, val, item_index, config):
    best = None
    for alpha, topsis_map in topsis_candidates.items():
        for rl_weight in config.rl_weights:
            topsis_weight = 1.0 - rl_weight
            scores = normalize_score_map(blend_scores(rl_scores_map, rl_weight, topsis_map, topsis_weight, user_ids))
            metrics = evaluate_scores(scores, val, item_index, config.top_k)
            if best is None or metrics["summary"]["NDCG@K"] > best["metrics"]["summary"]["NDCG@K"]:
                best = {
                    "alpha": alpha,
                    "rl_weight": rl_weight,
                    "topsis_weight": topsis_weight,
                    "scores": scores,
                    "metrics": metrics,
                }
    return best


def select_best_full_hybrid(cf_scores_map, rl_scores_map, topsis_candidates, user_ids, val, item_index, config):
    best = None
    for alpha, topsis_map in topsis_candidates.items():
        for rl_weight in config.rl_weights:
            for cf_weight in config.cf_weights:
                if rl_weight + cf_weight > 1.0:
                    continue
                topsis_weight = 1.0 - rl_weight - cf_weight
                scores = normalize_score_map(
                    blend_three_scores(cf_scores_map, cf_weight, rl_scores_map, rl_weight, topsis_map, topsis_weight, user_ids)
                )
                metrics = evaluate_scores(scores, val, item_index, config.top_k)
                if best is None or metrics["summary"]["NDCG@K"] > best["metrics"]["summary"]["NDCG@K"]:
                    best = {
                        "alpha": alpha,
                        "cf_weight": cf_weight,
                        "rl_weight": rl_weight,
                        "topsis_weight": topsis_weight,
                        "scores": scores,
                        "metrics": metrics,
                    }
    return best


def summarize_metric_across_splits(split_results, metric_name):
    values = [result[metric_name] for result in split_results]
    return {"mean": statistics.mean(values), "std": statistics.pstdev(values), "values": values}


def wilcoxon_test(split_table, challenger, baseline, metric_name):
    challenger_values = [split[metric_name] for split in split_table[challenger]]
    baseline_values = [split[metric_name] for split in split_table[baseline]]
    if challenger_values == baseline_values:
        return {"statistic": 0.0, "pvalue": 1.0, "rank_biserial": 0.0}
    statistic, pvalue = wilcoxon(challenger_values, baseline_values, alternative="two-sided")
    diffs = np.asarray(challenger_values, dtype=float) - np.asarray(baseline_values, dtype=float)
    nonzero = diffs[np.abs(diffs) > 1e-12]
    if len(nonzero) == 0:
        rank_biserial = 0.0
    else:
        ranks = rankdata(np.abs(nonzero), method="average")
        positive = ranks[nonzero > 0].sum()
        negative = ranks[nonzero < 0].sum()
        denom = positive + negative
        rank_biserial = 0.0 if denom == 0 else float((positive - negative) / denom)
    return {"statistic": float(statistic), "pvalue": float(pvalue), "rank_biserial": rank_biserial}


def run_split(seed, mode, items, histories, config):
    item_ids, item_index, family_lookup, static_lookup = build_indices(items)
    user_ids = sorted(histories)
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    train, val, test = make_split(histories, mode=mode, seed=seed)
    popularity = build_popularity(train, item_ids, item_index)
    interaction_matrix = build_interaction_matrix(train, user_ids, user_index, item_index)
    feature_matrix, feature_names = build_feature_matrix(train, item_ids, item_index, static_lookup)
    global_weights = compute_entropy_weights(feature_matrix)
    user_weights = build_user_topsis_weights(train, user_ids, item_index, feature_matrix)
    topsis_candidates = {
        alpha: normalize_score_map(topsis_scores(feature_matrix, user_ids, user_weights, global_weights, alpha))
        for alpha in config.topsis_alphas
    }

    repeat_last_map = build_repeat_last_scores(train, user_ids, item_ids, item_index, popularity)
    markov_map = build_item_markov_scores(train, user_ids, item_ids, item_index, popularity)
    cf_map = transition_cf_scores(train, user_ids, item_ids, item_index, popularity)
    nmf_result = fit_best_nmf(interaction_matrix, user_ids, user_index, val, item_index, config.top_k, config.nmf_components_grid, seed)
    svd_result = fit_best_svd(interaction_matrix, user_ids, user_index, val, item_index, config.top_k, config.svd_components_grid, seed)
    gru_result = fit_best_gru4rec(train, user_ids, val, item_index, config.top_k, config.gru_hidden_dims_grid, config, seed)
    sasrec_result = fit_best_sasrec(train, user_ids, val, item_index, config.top_k, config.sasrec_hidden_dims_grid, config, seed)
    rl_map = build_family_transition_bandit(
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

    best_topsis = select_best_topsis(user_ids, user_weights, global_weights, feature_matrix, val, item_index, config)
    best_cf_topsis = select_best_cf_topsis(cf_map, topsis_candidates, user_ids, val, item_index, config)
    best_rl_topsis = select_best_rl_topsis(rl_map, topsis_candidates, user_ids, val, item_index, config)
    best_full = select_best_full_hybrid(cf_map, rl_map, topsis_candidates, user_ids, val, item_index, config)

    model_scores = {
        "popularity": {user_id: popularity.copy() for user_id in user_ids},
        "repeat_last": repeat_last_map,
        "item_markov": markov_map,
        "transition_cf": cf_map,
        "nmf": nmf_result["scores"],
        "svd": svd_result["scores"],
        "gru4rec": gru_result["scores"],
        "sasrec": sasrec_result["scores"],
        "topsis": best_topsis["scores"],
        "rl_bandit": rl_map,
        "hybrid_cf_topsis": best_cf_topsis["scores"],
        "hybrid_rl_topsis": best_rl_topsis["scores"],
        "full_hybrid": best_full["scores"],
    }

    summary = {}
    for model_name, score_map in model_scores.items():
        summary[model_name] = evaluate_scores(score_map, test, item_index, config.top_k)["summary"]

    tuned_params = {
        "feature_names": feature_names,
        "global_entropy_weights": dict(zip(feature_names, [float(x) for x in global_weights])),
        "nmf_components": nmf_result["n_components"],
        "svd_components": svd_result["n_components"],
        "gru_hidden_dim": gru_result["hidden_dim"],
        "sasrec_hidden_dim": sasrec_result["hidden_dim"],
        "topsis_alpha": best_topsis["alpha"],
        "hybrid_cf_topsis": {"alpha": best_cf_topsis["alpha"], "cf_weight": best_cf_topsis["cf_weight"], "topsis_weight": best_cf_topsis["topsis_weight"]},
        "hybrid_rl_topsis": {"alpha": best_rl_topsis["alpha"], "rl_weight": best_rl_topsis["rl_weight"], "topsis_weight": best_rl_topsis["topsis_weight"]},
        "full_hybrid": {"alpha": best_full["alpha"], "cf_weight": best_full["cf_weight"], "rl_weight": best_full["rl_weight"], "topsis_weight": best_full["topsis_weight"]},
    }
    return {"seed": seed, "mode": mode, "metrics": summary, "tuned_params": tuned_params}


def aggregate_results(split_outputs):
    model_names = list(split_outputs[0]["metrics"].keys())
    aggregated = {}
    per_model_metric_table = {model_name: [] for model_name in model_names}
    for split_output in split_outputs:
        for model_name, metrics in split_output["metrics"].items():
            per_model_metric_table[model_name].append(metrics)
    for model_name, split_metrics in per_model_metric_table.items():
        aggregated[model_name] = {}
        for metric_name in ["HR@K", "NDCG@K", "MRR@K", "Precision@K"]:
            aggregated[model_name][metric_name] = summarize_metric_across_splits(split_metrics, metric_name)
    stat_tests = {
        "full_hybrid_vs_repeat_last_ndcg": wilcoxon_test(per_model_metric_table, "full_hybrid", "repeat_last", "NDCG@K"),
        "full_hybrid_vs_item_markov_ndcg": wilcoxon_test(per_model_metric_table, "full_hybrid", "item_markov", "NDCG@K"),
        "full_hybrid_vs_transition_cf_ndcg": wilcoxon_test(
            per_model_metric_table, "full_hybrid", "transition_cf", "NDCG@K"
        ),
        "full_hybrid_vs_hybrid_cf_topsis_ndcg": wilcoxon_test(per_model_metric_table, "full_hybrid", "hybrid_cf_topsis", "NDCG@K"),
        "full_hybrid_vs_gru4rec_ndcg": wilcoxon_test(per_model_metric_table, "full_hybrid", "gru4rec", "NDCG@K"),
        "full_hybrid_vs_sasrec_ndcg": wilcoxon_test(per_model_metric_table, "full_hybrid", "sasrec", "NDCG@K"),
    }
    return aggregated, stat_tests


def print_report(name, aggregated_results, stat_tests):
    print("=" * 88)
    print(name)
    print("=" * 88)
    header = f"{'Model':<22} {'HR@K':>14} {'NDCG@K':>14} {'MRR@K':>14} {'P@K':>14}"
    print(header)
    print("-" * len(header))
    for model_name, metrics in aggregated_results.items():
        print(
            f"{model_name:<22} "
            f"{metrics['HR@K']['mean']:.4f}\u00b1{metrics['HR@K']['std']:.4f} "
            f"{metrics['NDCG@K']['mean']:.4f}\u00b1{metrics['NDCG@K']['std']:.4f} "
            f"{metrics['MRR@K']['mean']:.4f}\u00b1{metrics['MRR@K']['std']:.4f} "
            f"{metrics['Precision@K']['mean']:.4f}\u00b1{metrics['Precision@K']['std']:.4f}"
        )
    print("-" * len(header))
    for test_name, result in stat_tests.items():
        print(
            f"{test_name}: statistic={result['statistic']:.4f}, "
            f"p={result['pvalue']:.4f}, rbc={result['rank_biserial']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the talent recommendation hybrid.")
    default_dataset_dir = Path(__file__).resolve().parents[1] / "prepared_datasets" / "karrierewege_ict_benchmark_v1"
    parser.add_argument("--dataset-dir", default=str(default_dataset_dir))
    parser.add_argument("--output-path", default=str(default_dataset_dir / "results_repeated.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=100)
    args = parser.parse_args()

    config = Config(dataset_dir=args.dataset_dir, output_path=args.output_path, top_k=args.top_k, repeats=args.repeats, seed_start=args.seed_start)
    items, interactions, metadata = load_dataset(config.dataset_dir)
    histories = build_histories(interactions)

    canonical_output = run_split(config.canonical_seed, "canonical", items, histories, config)
    repeated_outputs = [run_split(config.seed_start + offset, "repeated", items, histories, config) for offset in range(config.repeats)]
    aggregated_results, stat_tests = aggregate_results(repeated_outputs)

    print("=" * 88)
    print("DATASET")
    print("=" * 88)
    print(
        f"users={metadata['stats']['filtered_users']} "
        f"items={metadata['stats']['filtered_items']} "
        f"interactions={metadata['stats']['filtered_interactions']} "
        f"avg_seq_len={metadata['stats']['avg_sequence_length']:.3f}"
    )
    print(f"family_distribution={metadata['stats']['family_distribution']}")

    canonical_aggregated = {
        model_name: {
            metric_name: {"mean": metric_value, "std": 0.0, "values": [metric_value]}
            for metric_name, metric_value in metrics.items()
        }
        for model_name, metrics in canonical_output["metrics"].items()
    }
    print_report("CANONICAL SPLIT", canonical_aggregated, {})
    print_report("REPEATED CHRONOLOGICAL SPLITS", aggregated_results, stat_tests)

    payload = {
        "config": config.__dict__,
        "dataset_metadata": metadata,
        "canonical": canonical_output,
        "repeated": {"splits": repeated_outputs, "aggregated_results": aggregated_results, "stat_tests": stat_tests},
    }
    with open(config.output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
