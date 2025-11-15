#!/usr/bin/env python3
"""Compare all models and generate comprehensive comparison report with visualizations.

This script:
1. Loads results from multiple model runs
2. Compares metrics across models
3. Generates comparison visualizations
4. Creates detailed comparison report
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.visualization.plots import Plotter
from src.utils.path_utils import ensure_dir

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_model_results(model_dir: Path) -> dict:
    """Load all results from a model directory."""
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    results = {
        "model_dir": str(model_dir),
        "model_name": model_dir.name
    }
    
    # Load config
    config_path = model_dir / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            results["config"] = yaml.safe_load(f)
            results["model_name"] = results["config"].get("model_name", model_dir.name)
            results["model_type"] = results["config"].get("model_type", "unknown")
    
    # Load metrics
    for split in ["train", "val", "test"]:
        metrics_path = model_dir / f"{split}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                results[f"{split}_metrics"] = json.load(f)
    
    # Load summary
    summary_path = model_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)
            results.update(summary)
    
    # Load LOSO results
    loso_summary_path = model_dir / "loso" / "summary.json"
    if loso_summary_path.exists():
        with open(loso_summary_path, "r") as f:
            results["loso_summary"] = json.load(f)
    
    loso_stations_path = model_dir / "loso" / "station_metrics.csv"
    if loso_stations_path.exists():
        results["loso_stations"] = pd.read_csv(loso_stations_path)
    
    return results


def create_comparison_table(model_results: list, output_path: Path):
    """Create comparison table."""
    comparison_data = []
    
    for result in model_results:
        row = {
            "model_name": result.get("model_name", "unknown"),
            "model_type": result.get("model_type", "unknown")
        }
        
        # Test metrics
        test_metrics = result.get("test_metrics", {})
        if test_metrics:
            row.update({
                "test_mae": test_metrics.get("mae", np.nan),
                "test_rmse": test_metrics.get("rmse", np.nan),
                "test_r2": test_metrics.get("r2", np.nan),
                "test_mape": test_metrics.get("mape", np.nan)
            })
        
        # LOSO summary
        loso = result.get("loso_summary", {})
        if loso:
            row.update({
                "loso_mean_mae": loso.get("mean_mae", np.nan),
                "loso_std_mae": loso.get("std_mae", np.nan),
                "loso_mean_rmse": loso.get("mean_rmse", np.nan),
                "loso_mean_r2": loso.get("mean_r2", np.nan)
            })
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_path, index=False)
    
    return comparison_df


def create_comparison_visualizations(model_results: list, output_dir: Path):
    """Create comparison visualizations."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping visualizations")
        return
    
    plots_dir = output_dir / "plots"
    ensure_dir(plots_dir)
    
    # Prepare data
    model_names = [r.get("model_name", "unknown") for r in model_results]
    
    # Metrics comparison
    metrics_to_plot = ["mae", "rmse", "r2"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5*len(metrics_to_plot), 6))
    if len(metrics_to_plot) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics_to_plot):
        test_values = [r.get("test_metrics", {}).get(metric, np.nan) for r in model_results]
        
        axes[idx].bar(model_names, test_values)
        axes[idx].set_title(f"Test {metric.upper()}")
        axes[idx].set_ylabel("Value")
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plots_dir / 'metrics_comparison.png'}")
    
    # LOSO comparison (if available)
    loso_results = [r for r in model_results if "loso_summary" in r]
    if len(loso_results) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        loso_names = [r.get("model_name", "unknown") for r in loso_results]
        loso_mae_means = [r["loso_summary"]["mean_mae"] for r in loso_results]
        loso_mae_stds = [r["loso_summary"]["std_mae"] for r in loso_results]
        loso_r2_means = [r["loso_summary"]["mean_r2"] for r in loso_results]
        
        # MAE with error bars
        axes[0].bar(loso_names, loso_mae_means, yerr=loso_mae_stds, capsize=5)
        axes[0].set_title("LOSO Mean MAE (with std)")
        axes[0].set_ylabel("MAE (°C)")
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # R²
        axes[1].bar(loso_names, loso_r2_means)
        axes[1].set_title("LOSO Mean R²")
        axes[1].set_ylabel("R²")
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "loso_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plots_dir / 'loso_comparison.png'}")
    
    # Station-level LOSO comparison
    if len(loso_results) > 1:
        # Combine station metrics
        station_comparison = []
        for result in loso_results:
            if "loso_stations" in result:
                df = result["loso_stations"].copy()
                df["model_name"] = result.get("model_name", "unknown")
                station_comparison.append(df)
        
        if station_comparison:
            combined = pd.concat(station_comparison, ignore_index=True)
            
            # Box plot by model
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            sns.boxplot(data=combined, x="model_name", y="mae", ax=axes[0])
            axes[0].set_title("MAE Distribution by Model (LOSO)")
            axes[0].set_ylabel("MAE (°C)")
            axes[0].tick_params(axis='x', rotation=45)
            
            sns.boxplot(data=combined, x="model_name", y="r2", ax=axes[1])
            axes[1].set_title("R² Distribution by Model (LOSO)")
            axes[1].set_ylabel("R²")
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "loso_station_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {plots_dir / 'loso_station_distribution.png'}")


def generate_comparison_report(model_results: list, comparison_df: pd.DataFrame, output_dir: Path):
    """Generate markdown comparison report."""
    report_path = output_dir / "comparison_report.md"
    
    lines = []
    lines.append("# Model Comparison Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Models Compared**: {len(model_results)}")
    lines.append("")
    
    # Summary table
    lines.append("## Test Set Performance")
    lines.append("")
    lines.append(comparison_df[["model_name", "model_type", "test_mae", "test_rmse", "test_r2"]].to_markdown(index=False))
    lines.append("")
    
    # Best model
    best_mae_idx = comparison_df["test_mae"].idxmin()
    best_model = comparison_df.loc[best_mae_idx]
    lines.append(f"**Best Model (MAE)**: {best_model['model_name']} ({best_model['test_mae']:.4f}°C)")
    lines.append("")
    
    # LOSO summary
    loso_results = [r for r in model_results if "loso_summary" in r]
    if loso_results:
        lines.append("## LOSO (Leave-One-Station-Out) Performance")
        lines.append("")
        loso_data = []
        for r in loso_results:
            loso = r["loso_summary"]
            loso_data.append({
                "model": r.get("model_name", "unknown"),
                "mean_mae": f"{loso['mean_mae']:.4f} ± {loso['std_mae']:.4f}",
                "mean_rmse": f"{loso['mean_rmse']:.4f} ± {loso['std_rmse']:.4f}",
                "mean_r2": f"{loso['mean_r2']:.4f} ± {loso['std_r2']:.4f}"
            })
        loso_df = pd.DataFrame(loso_data)
        lines.append(loso_df.to_markdown(index=False))
        lines.append("")
    
    # Model details
    lines.append("## Model Details")
    lines.append("")
    for result in model_results:
        lines.append(f"### {result.get('model_name', 'Unknown')}")
        lines.append("")
        lines.append(f"- **Type**: {result.get('model_type', 'unknown')}")
        lines.append(f"- **Directory**: `{result.get('model_dir', 'unknown')}`")
        lines.append("")
        
        test_metrics = result.get("test_metrics", {})
        if test_metrics:
            lines.append("**Test Metrics**:")
            lines.append(f"- MAE: {test_metrics.get('mae', 'N/A'):.4f}°C")
            lines.append(f"- RMSE: {test_metrics.get('rmse', 'N/A'):.4f}°C")
            lines.append(f"- R²: {test_metrics.get('r2', 'N/A'):.4f}")
            lines.append("")
    
    # Visualizations
    lines.append("## Visualizations")
    lines.append("")
    lines.append("- `plots/metrics_comparison.png`: Test set metrics comparison")
    if loso_results:
        lines.append("- `plots/loso_comparison.png`: LOSO performance comparison")
        lines.append("- `plots/loso_station_distribution.png`: Station-level LOSO distribution")
    lines.append("")
    
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {report_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Compare multiple models")
    parser.add_argument(
        "model_dirs",
        type=Path,
        nargs="+",
        help="Paths to model directories to compare"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for comparison results"
    )
    
    args = parser.parse_args()
    
    # Load all model results
    print("Loading model results...")
    model_results = []
    for model_dir in args.model_dirs:
        try:
            result = load_model_results(Path(model_dir))
            model_results.append(result)
            print(f"  ✓ {result.get('model_name', model_dir.name)}")
        except Exception as e:
            print(f"  ✗ {model_dir}: {e}")
            continue
    
    if len(model_results) < 2:
        raise ValueError("Need at least 2 models to compare")
    
    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "experiments" / "comparisons" / f"comparison_{timestamp}"
    
    ensure_dir(output_dir)
    print(f"\nOutput directory: {output_dir}")
    
    # Create comparison table
    print("\nCreating comparison table...")
    comparison_df = create_comparison_table(
        model_results,
        output_dir / "comparison_table.csv"
    )
    print(comparison_df.to_string(index=False))
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_comparison_visualizations(model_results, output_dir)
    
    # Generate report
    print("\nGenerating comparison report...")
    generate_comparison_report(model_results, comparison_df, output_dir)
    
    # Save summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_models": len(model_results),
        "models": [r.get("model_name") for r in model_results],
        "comparison_table": comparison_df.to_dict(orient="records")
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("Comparison completed!")
    print(f"Results saved to: {output_dir}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

