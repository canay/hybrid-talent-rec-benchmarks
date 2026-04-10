import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


PARQUET_URLS = [
    "https://huggingface.co/api/datasets/ElenaSenger/Karrierewege/parquet/default/train/0.parquet",
    "https://huggingface.co/api/datasets/ElenaSenger/Karrierewege/parquet/default/train/1.parquet",
    "https://huggingface.co/api/datasets/ElenaSenger/Karrierewege/parquet/default/train/2.parquet",
    "https://huggingface.co/api/datasets/ElenaSenger/Karrierewege/parquet/default/train/3.parquet",
    "https://huggingface.co/api/datasets/ElenaSenger/Karrierewege/parquet/default/train/4.parquet",
    "https://huggingface.co/api/datasets/ElenaSenger/Karrierewege/parquet/default/train/5.parquet",
    "https://huggingface.co/api/datasets/ElenaSenger/Karrierewege/parquet/default/train/6.parquet",
    "https://huggingface.co/api/datasets/ElenaSenger/Karrierewege/parquet/default/train/7.parquet",
    "https://huggingface.co/api/datasets/ElenaSenger/Karrierewege/parquet/default/train/8.parquet",
    "https://huggingface.co/api/datasets/ElenaSenger/Karrierewege/parquet/default/validation/0.parquet",
    "https://huggingface.co/api/datasets/ElenaSenger/Karrierewege/parquet/default/test/0.parquet",
    "https://huggingface.co/api/datasets/ElenaSenger/Karrierewege/parquet/default/test/1.parquet",
]


OCCUPATION_FAMILY_MAP = {
    "application engineer": "software_engineering",
    "artificial intelligence engineer": "data_ai",
    "automation engineer": "hardware_automation",
    "blockchain developer": "software_engineering",
    "business intelligence manager": "data_ai",
    "chief ict security officer": "infrastructure_security",
    "cloud architect": "infrastructure_security",
    "cloud devops engineer": "infrastructure_security",
    "computer hardware engineer": "hardware_automation",
    "computer hardware engineering technician": "hardware_automation",
    "computer hardware test technician": "hardware_automation",
    "computer scientist": "data_ai",
    "cybersecurity risk manager": "infrastructure_security",
    "data analyst": "data_ai",
    "data centre operator": "infrastructure_security",
    "data engineer": "data_ai",
    "data scientist": "data_ai",
    "data warehouse designer": "data_ai",
    "database integrator": "infrastructure_security",
    "digital games designer": "digital_experience",
    "digital games developer": "digital_experience",
    "digital media designer": "digital_experience",
    "digital transformation manager": "tech_management",
    "e-learning developer": "digital_experience",
    "embedded systems software developer": "hardware_automation",
    "enterprise architect": "tech_management",
    "geographic information systems specialist": "data_ai",
    "green ict consultant": "tech_management",
    "ict account manager": "tech_management",
    "ict application configurator": "software_engineering",
    "ict application developer": "software_engineering",
    "ict business development manager": "tech_management",
    "ict capacity planner": "tech_management",
    "ict consultant": "tech_management",
    "ict help desk agent": "infrastructure_security",
    "ict help desk manager": "infrastructure_security",
    "ict information and knowledge manager": "tech_management",
    "ict network administrator": "infrastructure_security",
    "ict network technician": "infrastructure_security",
    "ict operations manager": "tech_management",
    "ict project manager": "tech_management",
    "ict quality assurance manager": "tech_management",
    "ict research manager": "tech_management",
    "ict system administrator": "infrastructure_security",
    "ict system analyst": "software_engineering",
    "ict system developer": "software_engineering",
    "ict system tester": "software_engineering",
    "ict technician": "infrastructure_security",
    "ict test analyst": "software_engineering",
    "ict trainer": "tech_management",
    "iot developer": "hardware_automation",
    "robotics engineer": "hardware_automation",
    "software architect": "software_engineering",
    "software developer": "software_engineering",
    "software tester": "software_engineering",
    "web content manager": "digital_experience",
    "web designer": "digital_experience",
    "web developer": "software_engineering",
}


DIGITAL_SKILL_KEYWORDS = (
    "software",
    "program",
    "coding",
    "developer",
    "web",
    "database",
    "data",
    "cloud",
    "devops",
    "security",
    "cyber",
    "network",
    "system",
    "ict",
    "digital",
    "ai",
    "machine learning",
    "analytics",
    "robot",
    "automation",
    "embedded",
    "iot",
    "blockchain",
)

INNOVATION_KEYWORDS = (
    "ai",
    "artificial intelligence",
    "machine learning",
    "cloud",
    "devops",
    "cyber",
    "security",
    "robot",
    "automation",
    "embedded",
    "iot",
    "blockchain",
    "data",
)


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def parse_skill_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [chunk.strip() for chunk in text.split("|") if chunk.strip()]


def digital_skill_density(skills):
    if not skills:
        return 0.0
    hits = 0
    for skill in skills:
        skill_text = skill.lower()
        if any(keyword in skill_text for keyword in DIGITAL_SKILL_KEYWORDS):
            hits += 1
    return hits / len(skills)


def innovation_score(label, skills):
    text = " ".join([label] + skills).lower()
    hits = sum(1 for keyword in INNOVATION_KEYWORDS if keyword in text)
    return min(1.0, hits / 4.0)


def role_level_score(label):
    lowered = label.lower()
    if any(keyword in lowered for keyword in ("chief", "head", "director")):
        return 1.0
    if any(keyword in lowered for keyword in ("manager", "architect", "lead")):
        return 0.8
    if any(keyword in lowered for keyword in ("consultant", "scientist")):
        return 0.6
    if any(keyword in lowered for keyword in ("engineer", "developer", "analyst", "administrator")):
        return 0.45
    if any(keyword in lowered for keyword in ("technician", "operator", "designer", "trainer")):
        return 0.3
    return 0.2


def load_filtered_sequences():
    frames = []
    for url in PARQUET_URLS:
        frame = pd.read_parquet(
            url,
            columns=["_id", "experience_order", "preferredLabel_en", "description_en", "skills"],
        )
        frame["preferredLabel_en"] = frame["preferredLabel_en"].astype(str).str.strip().str.lower()
        frame = frame[frame["preferredLabel_en"].isin(OCCUPATION_FAMILY_MAP)]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data = data.rename(columns={"_id": "user_id", "preferredLabel_en": "label"})
    data["user_id"] = data["user_id"].astype(str)
    data = data.sort_values(["user_id", "experience_order"]).reset_index(drop=True)
    return data


def iterative_sequence_k_core(data, min_user_sequence_length, min_item_user_support):
    filtered = data.copy()
    changed = True
    while changed:
        changed = False
        user_lengths = filtered.groupby("user_id").size()
        good_users = set(user_lengths[user_lengths >= min_user_sequence_length].index)
        if len(good_users) != user_lengths.shape[0]:
            filtered = filtered[filtered["user_id"].isin(good_users)].copy()
            changed = True

        item_support = filtered.groupby("label")["user_id"].nunique()
        good_labels = set(item_support[item_support >= min_item_user_support].index)
        if len(good_labels) != item_support.shape[0]:
            filtered = filtered[filtered["label"].isin(good_labels)].copy()
            changed = True

    filtered = filtered.sort_values(["user_id", "experience_order"]).reset_index(drop=True)
    filtered["sequence_pos"] = filtered.groupby("user_id").cumcount()
    return filtered


def build_item_table(filtered):
    item_rows = []
    item_support = filtered.groupby("label")["user_id"].nunique()
    interaction_counts = filtered["label"].value_counts()

    for label in sorted(filtered["label"].unique()):
        group = filtered[filtered["label"] == label]
        description = (
            group["description_en"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().mode().iloc[0]
        )
        skills = parse_skill_list(group["skills"].dropna().astype(str).iloc[0])
        item_rows.append(
            {
                "item_id": slugify(label),
                "label": label,
                "family": OCCUPATION_FAMILY_MAP[label],
                "description_en": description,
                "skills_json": json.dumps(skills, ensure_ascii=True),
                "skill_count": len(skills),
                "digital_skill_density": digital_skill_density(skills),
                "innovation_score": innovation_score(label, skills),
                "role_level_score": role_level_score(label),
                "interaction_count": int(interaction_counts[label]),
                "user_support": int(item_support[label]),
            }
        )
    return pd.DataFrame(item_rows).sort_values(["family", "label"]).reset_index(drop=True)


def build_interaction_table(filtered, items):
    item_id_lookup = dict(zip(items["label"], items["item_id"]))
    family_lookup = dict(zip(items["label"], items["family"]))
    interactions = filtered[["user_id", "sequence_pos", "label"]].copy()
    interactions["item_id"] = interactions["label"].map(item_id_lookup)
    interactions["family"] = interactions["label"].map(family_lookup)
    return interactions[["user_id", "sequence_pos", "item_id", "label", "family"]]


def build_canonical_split(interactions):
    rows = []
    for user_id, group in interactions.groupby("user_id"):
        group = group.sort_values("sequence_pos")
        sequence = group["item_id"].tolist()
        rows.append(
            {
                "user_id": user_id,
                "train_length": len(sequence) - 2,
                "val_position": len(sequence) - 2,
                "test_position": len(sequence) - 1,
                "val_item_id": sequence[-2],
                "test_item_id": sequence[-1],
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare the KARRIEREWEGE ICT benchmark.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "prepared_datasets" / "karrierewege_ict_benchmark_v1"),
    )
    parser.add_argument("--min-user-sequence-length", type=int, default=3)
    parser.add_argument("--min-item-user-support", type=int, default=25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_filtered_sequences()
    filtered = iterative_sequence_k_core(
        raw,
        min_user_sequence_length=args.min_user_sequence_length,
        min_item_user_support=args.min_item_user_support,
    )
    items = build_item_table(filtered)
    interactions = build_interaction_table(filtered, items)
    canonical_split = build_canonical_split(interactions)

    occupation_allowlist = pd.DataFrame(
        [{"label": label, "family": family, "selected_for_scan": True} for label, family in sorted(OCCUPATION_FAMILY_MAP.items())]
    )
    repeated_split_config = {
        "mode": "chronological_random_cut",
        "seed_start": 100,
        "repeats": 10,
        "rule": "For each user, choose test_index uniformly from [2, len(sequence)-1]; validation is test_index-1 and train is the prefix before validation.",
    }
    metadata = {
        "dataset_name": "karrierewege_ict_benchmark_v1",
        "source_name": "KARRIEREWEGE",
        "source_parquet_urls": PARQUET_URLS,
        "created_on": "2026-03-31",
        "task": "next_occupation_recommendation",
        "domain_scope": "skills-aware technology talent recommendation",
        "filtering": {
            "manual_ict_scan_size": len(OCCUPATION_FAMILY_MAP),
            "min_user_sequence_length": args.min_user_sequence_length,
            "min_item_user_support": args.min_item_user_support,
            "collapse_consecutive_duplicates": False,
        },
        "stats": {
            "raw_manual_filter_interactions": int(len(raw)),
            "raw_manual_filter_users": int(raw["user_id"].nunique()),
            "raw_manual_filter_items": int(raw["label"].nunique()),
            "filtered_interactions": int(len(interactions)),
            "filtered_users": int(interactions["user_id"].nunique()),
            "filtered_items": int(interactions["item_id"].nunique()),
            "avg_sequence_length": float(interactions.groupby("user_id").size().mean()),
            "max_sequence_length": int(interactions.groupby("user_id").size().max()),
            "min_sequence_length": int(interactions.groupby("user_id").size().min()),
            "family_distribution": dict(Counter(items["family"])),
        },
    }

    items.to_csv(output_dir / "items.csv", index=False)
    interactions.to_csv(output_dir / "interactions.csv", index=False)
    canonical_split.to_csv(output_dir / "canonical_split.csv", index=False)
    occupation_allowlist.to_csv(output_dir / "occupation_allowlist.csv", index=False)
    with open(output_dir / "repeated_split_config.json", "w", encoding="utf-8") as handle:
        json.dump(repeated_split_config, handle, indent=2)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("=" * 78)
    print("KARRIEREWEGE ICT BENCHMARK PREP")
    print("=" * 78)
    print(f"Output directory: {output_dir}")
    print(
        f"Manual ICT filter: interactions={metadata['stats']['raw_manual_filter_interactions']}, "
        f"users={metadata['stats']['raw_manual_filter_users']}, "
        f"items={metadata['stats']['raw_manual_filter_items']}"
    )
    print(
        f"Final benchmark: interactions={metadata['stats']['filtered_interactions']}, "
        f"users={metadata['stats']['filtered_users']}, "
        f"items={metadata['stats']['filtered_items']}, "
        f"avg_seq_len={metadata['stats']['avg_sequence_length']:.3f}"
    )
    print("Families:", dict(Counter(items["family"])))


if __name__ == "__main__":
    main()
