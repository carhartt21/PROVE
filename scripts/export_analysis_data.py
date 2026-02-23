#!/usr/bin/env python3
"""Export all PROVE analysis data to a target directory.

Copies all CSVs, JSONs, and metadata needed for the PROVE-Analysis repository.

Usage:
    python scripts/export_analysis_data.py /path/to/PROVE-Analysis/data
    python scripts/export_analysis_data.py /path/to/PROVE-Analysis/data --dry-run
    python scripts/export_analysis_data.py /path/to/PROVE-Analysis/data --include-stats
"""

import argparse
import json
import shutil
from pathlib import Path

# Resolve paths
PROVE_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_ROOT = Path("${AWARE_DATA_ROOT}")


def get_export_manifest(include_stats: bool = False, include_fid: bool = False) -> list[dict]:
    """Define all files to export with source → destination mapping."""
    
    manifest = []
    
    # ── Segmentation Results ──
    for name in [
        "downstream_results.csv",
        "downstream_results_stage2.csv",
        "downstream_results_cityscapes_gen.csv",
    ]:
        manifest.append({
            "src": PROVE_ROOT / name,
            "dst": "segmentation" / Path(name),
            "category": "segmentation",
            "description": f"Aggregated test results ({name})",
        })
    
    # ── Leaderboard Breakdowns ──
    breakdowns_dir = PROVE_ROOT / "result_figures" / "leaderboard" / "breakdowns"
    for csv_file in sorted(breakdowns_dir.glob("*.csv")):
        manifest.append({
            "src": csv_file,
            "dst": "leaderboard" / Path(csv_file.name),
            "category": "leaderboard",
            "description": f"Leaderboard breakdown: {csv_file.stem}",
        })
    
    # Leaderboard markdown files
    leaderboard_dir = PROVE_ROOT / "result_figures" / "leaderboard"
    for md_file in sorted(leaderboard_dir.glob("STRATEGY_LEADERBOARD_*.md")):
        manifest.append({
            "src": md_file,
            "dst": "leaderboard" / Path(md_file.name),
            "category": "leaderboard",
            "description": f"Leaderboard report: {md_file.stem}",
        })
    for md_file in sorted(leaderboard_dir.glob("DETAILED_GAINS_*.md")):
        manifest.append({
            "src": md_file,
            "dst": "leaderboard" / Path(md_file.name),
            "category": "leaderboard",
            "description": f"Gains analysis: {md_file.stem}",
        })
    for md_file in sorted((breakdowns_dir.parent).glob("*.md")):
        if md_file.name.startswith("DETAILED_GAINS_"):
            continue  # Already handled
        if md_file.name.startswith("STRATEGY_LEADERBOARD_"):
            continue  # Already handled
        manifest.append({
            "src": md_file,
            "dst": "leaderboard" / Path(md_file.name),
            "category": "leaderboard",
            "description": f"Analysis report: {md_file.stem}",
        })
    
    # ── Generative Quality ──
    quality_csv = PROVE_ROOT / "results" / "generative_quality" / "generative_quality.csv"
    manifest.append({
        "src": quality_csv,
        "dst": Path("quality") / "generative_quality.csv",
        "category": "quality",
        "description": "Composite quality scores (CQS, FID, LPIPS, SSIM, semantic metrics)",
    })
    
    # ── Ablation Studies ──
    ablation_files = {
        "ratio_ablation_full.csv": "Full ratio ablation results (ratio × strategy × dataset × model)",
        "ratio_ablation_consolidated.csv": "Consolidated ratio ablation summary",
        "extended_training_analysis.csv": "Extended training iteration convergence data",
    }
    for name, desc in ablation_files.items():
        src = PROVE_ROOT / "results" / name
        if src.exists():
            manifest.append({
                "src": src,
                "dst": Path("ablation") / name,
                "category": "ablation",
                "description": desc,
            })
    
    # Combination ablation
    combo_dir = PROVE_ROOT / "result_figures" / "combination_ablation"
    for csv_file in sorted(combo_dir.glob("*.csv")):
        manifest.append({
            "src": csv_file,
            "dst": Path("ablation") / f"combination_{csv_file.name}",
            "category": "ablation",
            "description": f"Combination ablation: {csv_file.stem}",
        })
    
    # ── Dataset Metadata ──
    split_stats = SCRATCH_ROOT / "FINAL_SPLITS" / "split_statistics.json"
    if split_stats.exists():
        manifest.append({
            "src": split_stats,
            "dst": Path("metadata") / "split_statistics.json",
            "category": "metadata",
            "description": "Dataset split statistics (6 datasets, 7 domains, 103K images)",
        })
    
    manifests_summary = SCRATCH_ROOT / "GENERATED_IMAGES" / "all_manifests_summary.json"
    if manifests_summary.exists():
        manifest.append({
            "src": manifests_summary,
            "dst": Path("metadata") / "all_manifests_summary.json",
            "category": "metadata",
            "description": "All generation methods summary (24 methods, counts, match rates)",
        })
    
    # Per-strategy manifests
    manifests_dir = PROVE_ROOT / "generated_manifests"
    for json_file in sorted(manifests_dir.glob("*_manifest.json")):
        manifest.append({
            "src": json_file,
            "dst": Path("metadata") / "manifests" / json_file.name,
            "category": "metadata",
            "description": f"Generation manifest: {json_file.stem}",
        })
    
    # ── Baseline Analysis CSVs ──
    baseline_dir = PROVE_ROOT / "result_figures" / "baseline_consolidated"
    for csv_file in sorted(baseline_dir.glob("*.csv")):
        manifest.append({
            "src": csv_file,
            "dst": Path("baseline") / csv_file.name,
            "category": "baseline",
            "description": f"Baseline analysis: {csv_file.stem}",
        })
    
    # ── Domain Analysis CSVs ──
    domain_dir = PROVE_ROOT / "result_figures" / "domain_adaptation"
    for csv_file in sorted(domain_dir.glob("*.csv")):
        manifest.append({
            "src": csv_file,
            "dst": Path("domain_analysis") / csv_file.name,
            "category": "domain",
            "description": f"Domain adaptation: {csv_file.stem}",
        })
    
    # Weather analysis
    weather_dir = PROVE_ROOT / "result_figures" / "weather_analysis_stage1"
    for csv_file in sorted(weather_dir.glob("*.csv")):
        manifest.append({
            "src": csv_file,
            "dst": Path("domain_analysis") / f"weather_{csv_file.name}",
            "category": "domain",
            "description": f"Weather domain analysis: {csv_file.stem}",
        })
    
    # Unified domain gap
    unified_dir = PROVE_ROOT / "result_figures" / "unified_domain_gap"
    for csv_file in sorted(unified_dir.glob("*.csv")):
        manifest.append({
            "src": csv_file,
            "dst": Path("domain_analysis") / f"unified_{csv_file.name}",
            "category": "domain",
            "description": f"Unified domain gap: {csv_file.stem}",
        })
    
    # ── Per-Strategy Quality Stats (optional, lightweight extraction) ──
    if include_stats:
        stats_root = SCRATCH_ROOT / "STATS"
        if stats_root.exists():
            dataset_names = {"ACDC", "BDD100k", "BDD10k", "OUTSIDE15k", "IDD-AW", "MapillaryVistas"}
            
            for method_dir in sorted(stats_root.iterdir()):
                if not method_dir.is_dir():
                    continue
                if method_dir.name in dataset_names:
                    continue
                
                # Mark stats files for lightweight extraction (handled in copy step)
                for stats_file in sorted(method_dir.glob("*_stats.json")):
                    manifest.append({
                        "src": stats_file,
                        "dst": Path("stats") / "methods" / method_dir.name / stats_file.name,
                        "category": "stats",
                        "description": f"Quality stats: {method_dir.name}/{stats_file.name}",
                        "extract_aggregate": True,  # Extract only aggregate metrics
                    })
    
    # ── FID Reference Files (optional, ~230MB) ──
    if include_fid:
        stats_root = SCRATCH_ROOT / "STATS"
        if stats_root.exists():
            for fid_file in sorted(stats_root.glob("*_fid.npz")):
                manifest.append({
                    "src": fid_file,
                    "dst": Path("stats") / "fid_references" / fid_file.name,
                    "category": "fid",
                    "description": f"FID reference: {fid_file.name}",
                })
    
    return manifest


def export_data(target_dir: Path, dry_run: bool = False, include_stats: bool = False, include_fid: bool = False):
    """Export all analysis data to target directory."""
    
    manifest = get_export_manifest(include_stats=include_stats, include_fid=include_fid)
    
    # Categorize and count
    categories = {}
    for item in manifest:
        cat = item["category"]
        categories.setdefault(cat, []).append(item)
    
    print("=" * 70)
    print(f"PROVE Analysis Data Export")
    print(f"=" * 70)
    print(f"Target directory: {target_dir}")
    print(f"Dry run: {dry_run}")
    print(f"Include per-strategy stats: {include_stats}")
    print()
    
    # Summary by category
    total_files = 0
    total_size = 0
    missing = []
    
    for cat, items in sorted(categories.items()):
        cat_size = 0
        cat_missing = 0
        for item in items:
            if item["src"].exists():
                if item.get("extract_aggregate"):
                    cat_size += 5 * 1024  # ~5KB per extracted stats file
                else:
                    cat_size += item["src"].stat().st_size
            else:
                cat_missing += 1
                missing.append(item)
        
        total_files += len(items)
        total_size += cat_size
        status = f" ({cat_missing} missing)" if cat_missing else ""
        print(f"  {cat:20s}: {len(items):3d} files, {cat_size / 1024 / 1024:.1f} MB{status}")
    
    print(f"\n  {'TOTAL':20s}: {total_files:3d} files, {total_size / 1024 / 1024:.1f} MB")
    
    if missing:
        print(f"\n⚠️  {len(missing)} files not found:")
        for item in missing[:10]:
            print(f"    - {item['src']}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    
    if dry_run:
        print("\n--- DRY RUN — no files copied ---")
        print(f"\nWould create directories:")
        dst_dirs = set()
        for item in manifest:
            dst_dirs.add((target_dir / item["dst"]).parent)
        for d in sorted(dst_dirs):
            print(f"  {d}")
        print(f"\nRun without --dry-run to copy files.")
        return
    
    # Actually copy files
    print(f"\nCopying files...")
    copied = 0
    skipped = 0
    errors = 0
    
    for item in manifest:
        src = item["src"]
        dst = target_dir / item["dst"]
        
        if not src.exists():
            skipped += 1
            continue
        
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            if item.get("extract_aggregate"):
                # Extract only aggregate metrics from large stats JSON
                with open(src) as f:
                    full_data = json.load(f)
                
                lightweight = {
                    "domain": full_data.get("domain"),
                    "generated": full_data.get("generated"),
                    "original": full_data.get("original"),
                    "num_pairs": full_data.get("num_pairs"),
                    "metrics": full_data.get("metrics", {}),
                }
                # Include semantic consistency summary but not per-image details
                if "semantic_consistency" in full_data:
                    sc = full_data["semantic_consistency"]
                    lightweight["semantic_consistency"] = {
                        k: v for k, v in sc.items() if k != "per_image_details"
                    }
                
                with open(dst, "w") as f:
                    json.dump(lightweight, f, indent=2)
            else:
                shutil.copy2(src, dst)
            
            copied += 1
        except Exception as e:
            print(f"  ERROR copying {src.name}: {e}")
            errors += 1
    
    print(f"\n✅ Copied: {copied} files")
    if skipped:
        print(f"⚠️  Skipped: {skipped} files (source not found)")
    if errors:
        print(f"❌ Errors: {errors} files")
    
    # Write export manifest JSON
    manifest_out = target_dir / "_export_manifest.json"
    manifest_data = {
        "exported_from": str(PROVE_ROOT),
        "exported_to": str(target_dir),
        "include_stats": include_stats,
        "total_files": copied,
        "categories": {cat: len(items) for cat, items in categories.items()},
        "files": [
            {
                "src": str(item["src"]),
                "dst": str(item["dst"]),
                "category": item["category"],
                "description": item["description"],
                "exists": item["src"].exists(),
            }
            for item in manifest
        ],
    }
    manifest_out.write_text(json.dumps(manifest_data, indent=2))
    print(f"\n📋 Export manifest written to: {manifest_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Export PROVE analysis data to a target directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/export_analysis_data.py ~/PROVE-Analysis/data --dry-run
  python scripts/export_analysis_data.py ~/PROVE-Analysis/data
  python scripts/export_analysis_data.py ~/PROVE-Analysis/data --include-stats
        """,
    )
    parser.add_argument("target_dir", type=Path, help="Target directory for exported data")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be copied")
    parser.add_argument(
        "--include-stats",
        action="store_true",
        help="Include per-strategy quality stats from /scratch (extracts aggregate metrics only, ~1MB)",
    )
    parser.add_argument(
        "--include-fid",
        action="store_true",
        help="Include FID reference npz files (~230MB total)",
    )
    
    args = parser.parse_args()
    export_data(
        args.target_dir,
        dry_run=args.dry_run,
        include_stats=args.include_stats,
        include_fid=args.include_fid,
    )


if __name__ == "__main__":
    main()
