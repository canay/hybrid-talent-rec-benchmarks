import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


CSV_URLS = [
    "https://huggingface.co/datasets/aida-ugent/JobHop/resolve/main/JobHop_train.csv?download=1",
    "https://huggingface.co/datasets/aida-ugent/JobHop/resolve/main/JobHop_val.csv?download=1",
    "https://huggingface.co/datasets/aida-ugent/JobHop/resolve/main/JobHop_test.csv?download=1",
]


ALLOWED_LABELS = {
    "ict help desk agent",
    "web developer",
    "software developer",
    "computer hardware repair technician",
    "ict network technician",
    "software analyst",
    "ict account manager",
    "ict technician",
    "ict system administrator",
    "ict network administrator",
    "ict project manager",
    "system configurator",
    "ict help desk manager",
    "data analyst",
    "ict application developer",
    "ict system analyst",
    "ict test analyst",
    "database integrator",
    "ict network engineer",
    "ict consultant",
    "ict product manager",
    "ict trainer",
    "data centre operator",
    "ict system integration consultant",
    "software tester",
    "ict business analyst",
    "ict system developer",
    "data warehouse designer",
    "database administrator",
    "mobile application developer",
    "ict change and configuration manager",
    "ict network architect",
    "embedded systems software developer",
    "user interface developer",
    "ict system architect",
    "e-learning developer",
    "ict system tester",
    "ict application configurator",
    "ict research consultant",
    "data quality specialist",
    "chief information officer",
    "database developer",
    "ict business development manager",
    "ict vendor relationship manager",
    "data scientist",
    "ict security manager",
    "it auditor",
    "integrated circuit design engineer",
    "ict information and knowledge manager",
    "ict disaster recovery analyst",
    "ict quality assurance manager",
    "cloud engineer",
    "enterprise architect",
    "embedded systems security engineer",
    "computer scientist",
    "ict security technician",
    "automation engineer",
    "digital forensics expert",
    "ict usability tester",
    "blockchain developer",
    "predictive maintenance expert",
    "ict resilience manager",
    "green ict consultant",
    "ict intelligent systems designer",
    "computer hardware engineering technician",
    "computer hardware engineer",
    "computer hardware test technician",
    "industrial mobile devices software developer",
    "big data archive librarian",
    "computer vision engineer",
    "blockchain architect",
    "chief data officer",
    "ict security engineer",
    "ict security administrator",
    "robotics engineer",
    "robotics engineering technician",
}


DIGITAL_KEYWORDS = (
    "software",
    "developer",
    "ict",
    "system",
    "network",
    "data",
    "database",
    "security",
    "cyber",
    "cloud",
    "web",
    "digital",
    "computer",
    "analytics",
    "automation",
    "robot",
    "embedded",
    "blockchain",
    "mobile",
    "interface",
    "architect",
)


INNOVATION_KEYWORDS = (
    "ai",
    "artificial intelligence",
    "machine learning",
    "cloud",
    "security",
    "cyber",
    "blockchain",
    "embedded",
    "robot",
    "automation",
    "data",
    "digital",
    "mobile",
)


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def quarter_to_int(value):
    text = str(value or "").strip().upper()
    match = re.match(r"Q([1-4])\s+(\d{4})", text)
    if not match:
        return -1
    quarter = int(match.group(1))
    year = int(match.group(2))
    return year * 10 + quarter


def assign_family(label):
    text = label.lower()
    if any(keyword in text for keyword in ("data ", "database", "big data", "computer vision", "data scientist")):
        return "data_ai"
    if any(keyword in text for keyword in ("security", "network", "help desk", "administrator", "resilience", "disaster recovery", "cloud")):
        return "infrastructure_security"
    if any(keyword in text for keyword in ("hardware", "embedded", "robot", "automation", "integrated circuit")):
        return "hardware_automation"
    if any(keyword in text for keyword in ("web", "interface", "e-learning")):
        return "digital_experience"
    if any(keyword in text for keyword in ("manager", "officer", "account", "consultant", "trainer", "auditor", "vendor", "chief", "product")):
        return "tech_management"
    return "software_engineering"


def proxy_skill_count(description):
    tokens = re.findall(r"[a-zA-Z]{4,}", str(description).lower())
    informative = {token for token in tokens if token not in {"that", "with", "from", "this", "their", "they", "them", "into", "such"}}
    return min(40, len(informative))


def digital_skill_density(label, description):
    text = f"{label} {description}".lower()
    hits = sum(text.count(keyword) for keyword in DIGITAL_KEYWORDS)
    token_count = max(20, len(re.findall(r"[a-zA-Z]+", text)))
    return min(1.0, hits / token_count * 10.0)


def innovation_score(label, description):
    text = f"{label} {description}".lower()
    hits = sum(1 for keyword in INNOVATION_KEYWORDS if keyword in text)
    return min(1.0, hits / 4.0)


def role_level_score(label):
    lowered = label.lower()
    if any(keyword in lowered for keyword in ("chief", "head")):
        return 1.0
    if any(keyword in lowered for keyword in ("manager", "architect", "officer")):
        return 0.8
    if any(keyword in lowered for keyword in ("consultant", "scientist", "auditor")):
        return 0.6
    if any(keyword in lowered for keyword in ("engineer", "developer", "analyst", "administrator")):
        return 0.45
    if any(keyword in lowered for keyword in ("technician", "tester", "trainer")):
        return 0.3
    return 0.2


def load_filtered_data():
    frames = []
    for url in CSV_URLS:
        frame = pd.read_csv(
            url,
            usecols=[
                "person_id",
                "matched_label",
                "matched_description",
                "matched_code",
                "start_date",
                "end_date",
                "university_studies",
            ],
        )
        frame["matched_label"] = frame["matched_label"].astype(str).str.strip().str.lower()
        frame = frame[frame["matched_label"].isin(ALLOWED_LABELS)]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["person_id", "matched_label", "start_date", "end_date"])
    data = data.rename(
        columns={
            "person_id": "user_id",
            "matched_label": "label",
            "matched_description": "description_en",
            "matched_code": "code",
        }
    )
    data["user_id"] = data["user_id"].astype(str)
    data["start_order"] = data["start_date"].map(quarter_to_int)
    data["end_order"] = data["end_date"].map(quarter_to_int)
    data = data.sort_values(["user_id", "start_order", "end_order", "label"]).reset_index(drop=True)
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

    filtered = filtered.sort_values(["user_id", "start_order", "end_order", "label"]).reset_index(drop=True)
    filtered["sequence_pos"] = filtered.groupby("user_id").cumcount()
    return filtered


def build_item_table(filtered):
    item_rows = []
    item_support = filtered.groupby("label")["user_id"].nunique()
    interaction_counts = filtered["label"].value_counts()
    university_rate = filtered.groupby("label")["university_studies"].mean()

    for label in sorted(filtered["label"].unique()):
        group = filtered[filtered["label"] == label]
        description = (
            group["description_en"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().mode().iloc[0]
        )
        code = group["code"].dropna().astype(str).mode().iloc[0]
        item_rows.append(
            {
                "item_id": slugify(label),
                "label": label,
                "family": assign_family(label),
                "description_en": description,
                "matched_code": code,
                "skill_count": proxy_skill_count(description),
                "digital_skill_density": digital_skill_density(label, description),
                "innovation_score": innovation_score(label, description),
                "role_level_score": role_level_score(label),
                "interaction_count": int(interaction_counts[label]),
                "user_support": int(item_support[label]),
                "university_rate": float(university_rate[label]),
            }
        )
    return pd.DataFrame(item_rows).sort_values(["family", "label"]).reset_index(drop=True)


def build_interaction_table(filtered, items):
    item_id_lookup = dict(zip(items["label"], items["item_id"]))
    family_lookup = dict(zip(items["label"], items["family"]))
    interactions = filtered[["user_id", "sequence_pos", "label", "start_date", "end_date", "university_studies"]].copy()
    interactions["item_id"] = interactions["label"].map(item_id_lookup)
    interactions["family"] = interactions["label"].map(family_lookup)
    return interactions[
        ["user_id", "sequence_pos", "item_id", "label", "family", "start_date", "end_date", "university_studies"]
    ]


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
    parser = argparse.ArgumentParser(description="Prepare the JobHop ICT benchmark.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "prepared_datasets" / "jobhop_ict_benchmark_v1"),
    )
    parser.add_argument("--min-user-sequence-length", type=int, default=3)
    parser.add_argument("--min-item-user-support", type=int, default=25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_filtered_data()
    filtered = iterative_sequence_k_core(
        raw,
        min_user_sequence_length=args.min_user_sequence_length,
        min_item_user_support=args.min_item_user_support,
    )
    items = build_item_table(filtered)
    interactions = build_interaction_table(filtered, items)
    canonical_split = build_canonical_split(interactions)

    allowlist = pd.DataFrame({"label": sorted(ALLOWED_LABELS)})
    repeated_split_config = {
        "mode": "chronological_random_cut",
        "seed_start": 100,
        "repeats": 10,
        "rule": "For each user, choose test_index uniformly from [2, len(sequence)-1]; validation is test_index-1 and train is the prefix before validation.",
    }
    metadata = {
        "dataset_name": "jobhop_ict_benchmark_v1",
        "source_name": "JobHop",
        "source_csv_urls": CSV_URLS,
        "created_on": "2026-03-31",
        "task": "next_occupation_recommendation",
        "domain_scope": "skills-aware technology talent recommendation",
        "filtering": {
            "manual_ict_scan_size": len(ALLOWED_LABELS),
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
    allowlist.to_csv(output_dir / "occupation_allowlist.csv", index=False)
    with open(output_dir / "repeated_split_config.json", "w", encoding="utf-8") as handle:
        json.dump(repeated_split_config, handle, indent=2)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("=" * 78)
    print("JOBHOP ICT BENCHMARK PREP")
    print("=" * 78)
    print(f"Output directory: {output_dir}")
    print(
        f"Manual ICT filter: interactions={metadata['stats']['raw_manual_filter_interactions']}, "
        f"users={metadata['stats']['raw_manual_filter_users']}, items={metadata['stats']['raw_manual_filter_items']}"
    )
    print(
        f"Final benchmark: interactions={metadata['stats']['filtered_interactions']}, "
        f"users={metadata['stats']['filtered_users']}, items={metadata['stats']['filtered_items']}, "
        f"avg_seq_len={metadata['stats']['avg_sequence_length']:.3f}"
    )
    print("Families:", dict(Counter(items["family"])))


if __name__ == "__main__":
    main()
