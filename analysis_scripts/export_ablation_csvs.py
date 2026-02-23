#!/usr/bin/env python3
"""Export all ablation study results to clean CSV files for the IEEE repo.

Usage:
    python analysis_scripts/export_ablation_csvs.py [--ieee-dir PATH]

Ablation studies exported:
    1. CS-Ratio (Cityscapes ratio ablation) — WEIGHTS_CITYSCAPES_RATIO
    2. S1-Ratio (Stage 1 ratio ablation) — WEIGHTS_STAGE1_RATIO
    3. Noise Ablation — WEIGHTS_NOISE_ABLATION
    4. Extended Training — WEIGHTS_EXTENDED_ABLATION
    5. Combination Ablation — WEIGHTS_COMBINATION_ABLATION
    6. From-Scratch (ratio=0.50 + ratio=0.00) — WEIGHTS_FROM_SCRATCH
    7. Loss Ablation — WEIGHTS_LOSS_ABLATION
"""
import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path


SCRATCH = Path("${AWARE_DATA_ROOT}")
LOCAL_OUTPUT = Path("analysis_scripts/result_figures/ablation_exports")
IEEE_DEFAULT = Path("${HOME}/repositories/-IEEE-Access-01-26-Data-Augmentation/analysis/data/ablation")

# Datasets that use native classes (not 19 Cityscapes classes)
NATIVE_CLASS_DATASETS = {"mapillaryvistas", "outside15k"}


def find_results(base_dir, pattern="**/results.json"):
    """Find all results.json files under a directory."""
    return sorted(base_dir.glob(pattern))


def load_result(path):
    """Load a results.json file and return overall metrics."""
    with open(path) as f:
        data = json.load(f)
    overall = data.get("overall", {})
    per_domain = data.get("per_domain", {})
    return overall, per_domain


def parse_path_components(result_path, base_dir):
    """Extract strategy/dataset/model/ratio from result path relative to base_dir."""
    rel = result_path.relative_to(base_dir)
    parts = list(rel.parts)
    # Typical: strategy/dataset/model_ratioXpYZ/test_results_detailed/timestamp/results.json
    # Or: strategy/dataset/model/test_results_detailed/timestamp/results.json
    info = {}

    # Find test_results_* directory
    trd_idx = None
    for i, p in enumerate(parts):
        if p.startswith("test_results"):
            trd_idx = i
            info["test_type"] = p.replace("test_results_", "")
            break
    if trd_idx is None:
        return None

    path_parts = parts[:trd_idx]

    if len(path_parts) >= 3:
        info["strategy"] = path_parts[0]
        info["dataset"] = path_parts[1]
        model_part = path_parts[2]
    elif len(path_parts) == 2:
        info["strategy"] = path_parts[0]
        info["dataset"] = ""
        model_part = path_parts[1]
    else:
        return None

    # Parse model and ratio from model_part
    # e.g., "segformer_mit-b3_ratio0p50" or "segformer_mit-b3"
    ratio_match = re.search(r"_ratio(\d+p\d+)", model_part)
    if ratio_match:
        ratio_str = ratio_match.group(1).replace("p", ".")
        info["ratio"] = float(ratio_str)
        info["model"] = model_part[: ratio_match.start()]
    else:
        info["ratio"] = None
        info["model"] = model_part

    # Timestamp
    if trd_idx + 1 < len(parts):
        info["timestamp"] = parts[trd_idx + 1]
    else:
        info["timestamp"] = ""

    return info


def parse_extended_path(result_path, base_dir):
    """Parse WEIGHTS_EXTENDED_ABLATION path structure.
    Structure: stage1/{strategy}/{dataset}/{model}/test_results_detailed/iter_XXXXX/{timestamp}/results.json
    Or: cityscapes_gen/{strategy}/{dataset}/{model}/test_results_detailed/iter_XXXXX/{timestamp}/results.json
    """
    rel = result_path.relative_to(base_dir)
    parts = list(rel.parts)

    trd_idx = None
    for i, p in enumerate(parts):
        if p.startswith("test_results"):
            trd_idx = i
            break
    if trd_idx is None:
        return None

    info = {"test_type": parts[trd_idx].replace("test_results_", "")}
    path_before = parts[:trd_idx]

    if len(path_before) >= 4:
        info["stage"] = path_before[0]  # stage1 or cityscapes_gen
        info["strategy"] = path_before[1]
        info["dataset"] = path_before[2]
        info["model"] = path_before[3]
    elif len(path_before) >= 3:
        info["stage"] = path_before[0]
        info["strategy"] = path_before[1]
        info["dataset"] = path_before[2]
        info["model"] = ""
    else:
        return None

    # After test_results_detailed: iter_XXXXX/timestamp/results.json
    after = parts[trd_idx + 1:]
    iter_match = None
    for p in after:
        m = re.match(r"iter_(\d+)", p)
        if m:
            iter_match = int(m.group(1))
            break
    info["iteration"] = iter_match

    info["timestamp"] = after[-2] if len(after) >= 2 else ""

    return info


def parse_ratio_path(result_path, base_dir):
    """Parse CS-Ratio or S1-Ratio path.
    CS-Ratio: {strategy}/{dataset}/{model}_ratio{X}p{Y}/test_results_detailed/...
    Also handles iter_XXXXX subdirs for multi-checkpoint tests.
    """
    rel = result_path.relative_to(base_dir)
    parts = list(rel.parts)

    trd_idx = None
    for i, p in enumerate(parts):
        if p.startswith("test_results"):
            trd_idx = i
            break
    if trd_idx is None:
        return None

    info = {"test_type": parts[trd_idx].replace("test_results_", "")}
    path_before = parts[:trd_idx]

    if len(path_before) >= 3:
        info["strategy"] = path_before[0]
        info["dataset"] = path_before[1]
        model_part = path_before[2]
    elif len(path_before) >= 2:
        info["strategy"] = path_before[0]
        model_part = path_before[1]
        info["dataset"] = ""
    else:
        return None

    # Parse ratio from model name
    ratio_match = re.search(r"_ratio(\d+p\d+)", model_part)
    if ratio_match:
        info["ratio"] = float(ratio_match.group(1).replace("p", "."))
        info["model"] = model_part[: ratio_match.start()]
    else:
        info["ratio"] = None
        info["model"] = model_part

    # Check for iteration subdirectory
    after = parts[trd_idx + 1:]
    iter_val = None
    for p in after:
        m = re.match(r"iter_(\d+)", p)
        if m:
            iter_val = int(m.group(1))
            break
    info["iteration"] = iter_val
    info["timestamp"] = after[-2] if len(after) >= 2 and after[-1] == "results.json" else ""

    return info


def parse_noise_path(result_path, base_dir):
    """Parse WEIGHTS_NOISE_ABLATION path.
    Structure: {strategy}/{dataset}/{model}_ratio{X}p{Y}/test_results_detailed/...
    """
    return parse_ratio_path(result_path, base_dir)


def export_study(name, base_dir, parse_fn, fields, output_path, extra_fields_fn=None):
    """Generic exporter for an ablation study."""
    results_files = find_results(base_dir)
    if not results_files:
        print(f"  {name}: 0 results found — skipping")
        return 0

    rows = []
    for rf in results_files:
        info = parse_fn(rf, base_dir)
        if info is None:
            continue
        overall, per_domain = load_result(rf)
        if not overall:
            continue

        row = {}
        for f in fields:
            row[f] = info.get(f, "")

        row["mIoU"] = round(overall.get("mIoU", 0), 2)
        row["mAcc"] = round(overall.get("mAcc", 0), 2)
        row["aAcc"] = round(overall.get("aAcc", 0), 2)
        row["fwIoU"] = round(overall.get("fwIoU", 0), 2)
        row["num_images"] = overall.get("num_images", 0)

        # Per-domain mIoU columns
        if per_domain:
            for domain_name, domain_data in sorted(per_domain.items()):
                summary = domain_data.get("summary", domain_data)
                domain_miou = summary.get("mIoU", None)
                if domain_miou is not None:
                    row[f"mIoU_{domain_name}"] = round(domain_miou, 2)

        if extra_fields_fn:
            extra_fields_fn(row, info, overall, per_domain)

        rows.append(row)

    if not rows:
        print(f"  {name}: 0 valid rows — skipping")
        return 0

    # Determine all columns
    all_cols = list(fields) + ["mIoU", "mAcc", "aAcc", "fwIoU", "num_images"]
    domain_cols = sorted(set(k for row in rows for k in row.keys() if k.startswith("mIoU_")))
    all_cols += domain_cols

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: tuple(str(r.get(c, "")) for c in fields)):
            writer.writerow(row)

    print(f"  {name}: {len(rows)} rows → {output_path.name}")
    return len(rows)


def export_cs_ratio(output_dir):
    """Export CS-Ratio ablation (WEIGHTS_CITYSCAPES_RATIO)."""
    base = SCRATCH / "WEIGHTS_CITYSCAPES_RATIO"
    if not base.exists():
        return 0
    return export_study(
        "CS-Ratio",
        base,
        parse_ratio_path,
        ["strategy", "dataset", "model", "ratio", "test_type"],
        output_dir / "cs_ratio_ablation.csv",
    )


def export_s1_ratio(output_dir):
    """Export S1-Ratio ablation (WEIGHTS_STAGE1_RATIO)."""
    base = SCRATCH / "WEIGHTS_STAGE1_RATIO"
    if not base.exists():
        return 0
    return export_study(
        "S1-Ratio",
        base,
        parse_ratio_path,
        ["strategy", "dataset", "model", "ratio", "test_type"],
        output_dir / "s1_ratio_ablation.csv",
    )


def export_noise(output_dir):
    """Export Noise ablation (WEIGHTS_NOISE_ABLATION)."""
    base = SCRATCH / "WEIGHTS_NOISE_ABLATION"
    if not base.exists():
        return 0
    return export_study(
        "Noise Ablation",
        base,
        parse_noise_path,
        ["strategy", "dataset", "model", "ratio", "test_type"],
        output_dir / "noise_ablation.csv",
    )


def export_extended(output_dir):
    """Export Extended Training ablation (WEIGHTS_EXTENDED_ABLATION)."""
    base = SCRATCH / "WEIGHTS_EXTENDED_ABLATION"
    if not base.exists():
        return 0
    return export_study(
        "Extended Training",
        base,
        parse_extended_path,
        ["stage", "strategy", "dataset", "model", "iteration", "test_type"],
        output_dir / "extended_training_ablation.csv",
    )


def export_combination(output_dir):
    """Export Combination ablation (WEIGHTS_COMBINATION_ABLATION)."""
    base = SCRATCH / "WEIGHTS_COMBINATION_ABLATION"
    if not base.exists():
        return 0
    return export_study(
        "Combination Ablation",
        base,
        parse_path_components,
        ["strategy", "dataset", "model", "test_type"],
        output_dir / "combination_ablation.csv",
    )


def export_from_scratch(output_dir):
    """Export From-Scratch results (WEIGHTS_FROM_SCRATCH) — both ratio=0.50 and ratio=0.00."""
    base = SCRATCH / "WEIGHTS_FROM_SCRATCH"
    if not base.exists():
        return 0
    return export_study(
        "From-Scratch",
        base,
        parse_path_components,
        ["strategy", "dataset", "model", "ratio", "test_type"],
        output_dir / "from_scratch_ablation.csv",
    )


def export_loss(output_dir):
    """Export Loss ablation (WEIGHTS_LOSS_ABLATION)."""
    base = SCRATCH / "WEIGHTS_LOSS_ABLATION"
    if not base.exists():
        return 0

    def parse_loss_path(result_path, base_dir):
        """Parse loss ablation paths.
        Structure: {loss_type}/{strategy}/{dataset}/{model}/test_results_*/...
        """
        rel = result_path.relative_to(base_dir)
        parts = list(rel.parts)
        trd_idx = None
        for i, p in enumerate(parts):
            if p.startswith("test_results"):
                trd_idx = i
                break
        if trd_idx is None:
            return None
        path_before = parts[:trd_idx]
        info = {"test_type": parts[trd_idx].replace("test_results_", "")}
        if len(path_before) >= 4:
            info["loss_type"] = path_before[0]
            info["strategy"] = path_before[1]
            info["dataset"] = path_before[2]
            info["model"] = path_before[3]
        elif len(path_before) >= 3:
            info["loss_type"] = path_before[0]
            info["strategy"] = path_before[1]
            info["dataset"] = path_before[2]
            info["model"] = ""
        else:
            return None
        return info

    return export_study(
        "Loss Ablation",
        base,
        parse_loss_path,
        ["loss_type", "strategy", "dataset", "model", "test_type"],
        output_dir / "loss_ablation.csv",
    )


def main():
    parser = argparse.ArgumentParser(description="Export all ablation study results to CSV")
    parser.add_argument(
        "--ieee-dir",
        type=Path,
        default=IEEE_DEFAULT,
        help="IEEE repo ablation directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LOCAL_OUTPUT,
        help="Local output directory",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Don't copy to IEEE repo",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Exporting Ablation Study CSVs")
    print("=" * 60)

    total = 0
    csv_files = []

    for name, fn in [
        ("CS-Ratio", export_cs_ratio),
        ("S1-Ratio", export_s1_ratio),
        ("Noise", export_noise),
        ("Extended Training", export_extended),
        ("Combination", export_combination),
        ("From-Scratch", export_from_scratch),
        ("Loss", export_loss),
    ]:
        count = fn(output_dir)
        total += count
        if count > 0:
            # Find the csv that was just written
            csvs = sorted(output_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
            if csvs:
                csv_files.append(csvs[-1])

    print(f"\nTotal: {total} rows across {len(csv_files)} CSV files")
    print(f"Local output: {output_dir}")

    # Copy to IEEE repo
    if not args.no_copy and args.ieee_dir.exists():
        print(f"\nCopying to IEEE repo: {args.ieee_dir}")
        for csv_file in csv_files:
            dest = args.ieee_dir / csv_file.name
            shutil.copy2(csv_file, dest)
            print(f"  → {csv_file.name}")
        print(f"Copied {len(csv_files)} CSV files to IEEE repo")
    elif not args.no_copy:
        print(f"\nIEEE repo dir not found: {args.ieee_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
