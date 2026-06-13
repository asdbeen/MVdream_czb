import hashlib
import json
import random
from pathlib import Path


CATEGORY_FIELDS = ["entity", "volume", "direction", "operation", "affect"]
FIELD_INDEX_LIMITS = {
    "entity": (0, 4),
    "volume": (0, 32),
    "direction": (0, 19),
    "operation": (0, 67),
    "affect": (0, 73),
}

REPO_ROOT = Path(__file__).resolve().parent.parent
MAPPING_PATH = REPO_ROOT / "1001 BUILDING FORMS_Dataset_rewrite_value_index.json"
DATA_ROOT = REPO_ROOT / "customized_simple_dataset_tagVersion_simplified" / "data"
OUTPUT_ROOT = REPO_ROOT / "Animation" / "randomizecat"


def load_mapping():
    with MAPPING_PATH.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    pools = {}
    for field in CATEGORY_FIELDS:
        min_index, max_index = FIELD_INDEX_LIMITS[field]
        allowed = []
        for label, index in raw[field].items():
            if min_index <= int(index) <= max_index:
                allowed.append((int(index), label))
        allowed.sort()
        pools[field] = [label for _, label in allowed]
    return pools


def load_category(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def build_rng(sample_id: str, replace_count: int) -> random.Random:
    seed_text = f"{sample_id}:{replace_count}"
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def choose_new_value(field: str, original_value: str, pools, rng: random.Random) -> str:
    candidates = [value for value in pools[field] if value != original_value]
    if not candidates:
        raise ValueError(f"No alternative values available for field '{field}'")
    return rng.choice(candidates)


def generate_variants():
    pools = load_mapping()
    category_paths = sorted(DATA_ROOT.glob("*/category.json"))
    summary = {
        "source_root": str(DATA_ROOT),
        "mapping_path": str(MAPPING_PATH),
        "output_root": str(OUTPUT_ROOT),
        "total_samples": len(category_paths),
        "variants": {},
    }

    for replace_count in range(1, 6):
        variant_name = f"replace_{replace_count}"
        written = 0

        for category_path in category_paths:
            sample_id = category_path.parent.name
            original = load_category(category_path)
            updated = dict(original)
            rng = build_rng(sample_id, replace_count)

            fields_to_replace = rng.sample(CATEGORY_FIELDS, replace_count)
            changes = []
            for field in fields_to_replace:
                old_value = str(original.get(field, "")).strip()
                new_value = choose_new_value(field, old_value, pools, rng)
                updated[field] = new_value
                changes.append(
                    {
                        "field": field,
                        "old": old_value,
                        "new": new_value,
                    }
                )

            sample_output_dir = OUTPUT_ROOT / variant_name / sample_id
            write_json(sample_output_dir / "category.json", updated)
            write_json(
                sample_output_dir / "changed_fields.json",
                {
                    "sample_id": sample_id,
                    "replace_count": replace_count,
                    "changed_count": len(changes),
                    "changed_fields": changes,
                },
            )
            written += 1

        summary["variants"][variant_name] = {"samples_written": written}

    write_json(OUTPUT_ROOT / "summary.json", summary)


if __name__ == "__main__":
    generate_variants()
