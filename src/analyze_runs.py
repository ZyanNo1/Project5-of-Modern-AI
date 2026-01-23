import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import pandas as pd


@dataclass
class RunMetrics:
    """Extended metrics including per-class F1 scores"""
    run_name: str
    run_dir: Path
    
    # Overall metrics
    internal_test_macro_f1: Optional[float]
    internal_test_accuracy: Optional[float]
    best_val_macro_f1: Optional[float]
    
    # Per-class metrics
    negative_f1: Optional[float]
    neutral_f1: Optional[float]
    positive_f1: Optional[float]
    
    # Hyperparameters
    hparams: Dict[str, Any]
    
    # Resource usage
    ckpt_mb: float


def load_per_class_metrics(run_dir: Path) -> Dict[str, Optional[float]]:
    """Load per-class F1 scores from internal_test_metrics.json"""
    metrics_path = run_dir / "eval" / "internal_test_metrics.json"
    
    if not metrics_path.exists():
        return {"negative_f1": None, "neutral_f1": None, "positive_f1": None}
    
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        # Extract per-class F1 from classification_report
        report = metrics.get("classification_report", {})
        
        return {
            "negative_f1": report.get("negative", {}).get("f1-score"),
            "neutral_f1": report.get("neutral", {}).get("f1-score"),
            "positive_f1": report.get("positive", {}).get("f1-score"),
        }
    except Exception as e:
        print(f"Warning: Cannot parse {metrics_path}: {e}")
        return {"negative_f1": None, "neutral_f1": None, "positive_f1": None}


def load_hparams(run_dir: Path) -> Dict[str, Any]:
    """Load hyperparameters from hparams.json"""
    hparams_path = run_dir / "hparams.json"
    if not hparams_path.exists():
        return {}
    
    try:
        with open(hparams_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def load_summary_csv(csv_path: Path) -> pd.DataFrame:
    """Load run_summary.csv into DataFrame"""
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")
    
    return pd.read_csv(csv_path)


def enrich_with_per_class_metrics(df: pd.DataFrame, outputs_dir: Path) -> pd.DataFrame:
    
    per_class_data = []
    for _, row in df.iterrows():
        run_dir = Path(row['run_dir'])
        metrics = load_per_class_metrics(run_dir)
        per_class_data.append(metrics)
    
    for key in ["negative_f1", "neutral_f1", "positive_f1"]:
        df[key] = [d[key] for d in per_class_data]
    
    return df


def print_top_k_runs(df: pd.DataFrame, metric: str, k: int = 10):
    """Print top K runs sorted by a metric"""
    print(f"\n{'='*80}")
    print(f"Top {k} Runs by {metric}")
    print('='*80)
    
    if metric not in df.columns:
        print(f"Error: Metric '{metric}' not found in summary")
        return
    
    # Sort and filter out NaN
    df_sorted = df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(k)
    
    # Print table
    cols = ["run_name", metric, "best_val_macro_f1", "fusion", "two_stage", 
            "lr_clip", "hidden_dim", "dropout"]
    
    # Filter columns that exist
    cols = [c for c in cols if c in df_sorted.columns]
    
    print(df_sorted[cols].to_string(index=False))
    print()


def print_per_class_analysis(df: pd.DataFrame):
    """Analyze per-class performance"""
    print(f"\n{'='*80}")
    print("Per-Class F1 Score Analysis")
    print('='*80)
    
    # Check if per-class metrics exist
    if "negative_f1" not in df.columns:
        print("Per-class metrics not available (run with --enrich)")
        return
    
    # Statistics per class
    for cls in ["negative", "neutral", "positive"]:
        col = f"{cls}_f1"
        if col not in df.columns:
            continue
        
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        
        print(f"\n{cls.capitalize()} Class:")
        print(f"  Mean F1:   {valid.mean():.4f}")
        print(f"  Std F1:    {valid.std():.4f}")
        print(f"  Max F1:    {valid.max():.4f} (run: {df.loc[df[col].idxmax(), 'run_name']})")
        print(f"  Min F1:    {valid.min():.4f}")
    
    # Best overall run
    print(f"\n{'='*80}")
    print("Best Run for Each Class:")
    print('='*80)
    
    for cls in ["negative", "neutral", "positive"]:
        col = f"{cls}_f1"
        if col not in df.columns or df[col].isna().all():
            continue
        
        best_idx = df[col].idxmax()
        best_run = df.loc[best_idx]
        
        print(f"\n{cls.capitalize()}: {best_run['run_name']}")
        print(f"  F1 Score: {best_run[col]:.4f}")
        print(f"  Macro F1: {best_run.get('internal_test_macro_f1', 'N/A')}")
        print(f"  Config:   fusion={best_run.get('fusion')}, lr_clip={best_run.get('lr_clip')}")


def print_grouped_statistics(df: pd.DataFrame, group_by: str):
    print(f"\n{'='*80}")
    print(f"Statistics Grouped by {group_by}")
    print('='*80)
    
    if group_by not in df.columns:
        print(f"Error: Column '{group_by}' not found")
        return
    
    grouped = df.groupby(group_by)["internal_test_macro_f1"].agg(['mean', 'std', 'max', 'count'])
    grouped = grouped.sort_values('mean', ascending=False)
    
    print(grouped.to_string())
    print()


def print_correlation_analysis(df: pd.DataFrame):
    print(f"\n{'='*80}")
    print("Hyperparameter vs Performance Correlation")
    print('='*80)
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    target = 'internal_test_macro_f1'
    
    if target not in numeric_cols:
        print("Target metric not available")
        return
    
    # Calculate correlations
    correlations = df[numeric_cols].corr()[target].drop(target).sort_values(ascending=False)
    
    print("\nTop Positive Correlations:")
    print(correlations.head(10).to_string())
    
    print("\nTop Negative Correlations:")
    print(correlations.tail(10).to_string())
    print()


def export_enhanced_csv(df: pd.DataFrame, output_path: Path):
    """Export enhanced summary with per-class metrics"""
    df.to_csv(output_path, index=False)
    print(f"Enhanced summary saved to: {output_path}")


def generate_latex_table(df: pd.DataFrame, output_path: Path, top_k: int = 10):
    """Generate LaTeX table for top K runs"""
    df_top = df.dropna(subset=['internal_test_macro_f1']).sort_values(
        'internal_test_macro_f1', ascending=False
    ).head(top_k)
    
    latex = "\\begin{table}[h]\n\\centering\n\\caption{Top 10 Model Configurations}\n"
    latex += "\\begin{tabular}{llcccccc}\n\\hline\n"
    latex += "Run & Fusion & Two-Stage & Test F1 & Neg F1 & Neu F1 & Pos F1 & Val F1 \\\\\n\\hline\n"
    
    for _, row in df_top.iterrows():
        run_short = row['run_name'].replace('run_', '')[:15]
        fusion = row.get('fusion', 'N/A')
        two_stage = 'Y' if row.get('two_stage') else 'N'
        test_f1 = f"{row['internal_test_macro_f1']:.3f}" if pd.notna(row['internal_test_macro_f1']) else '-'
        neg_f1 = f"{row.get('negative_f1', 0):.3f}" if pd.notna(row.get('negative_f1')) else '-'
        neu_f1 = f"{row.get('neutral_f1', 0):.3f}" if pd.notna(row.get('neutral_f1')) else '-'
        pos_f1 = f"{row.get('positive_f1', 0):.3f}" if pd.notna(row.get('positive_f1')) else '-'
        val_f1 = f"{row.get('best_val_macro_f1', 0):.3f}" if pd.notna(row.get('best_val_macro_f1')) else '-'
        
        latex += f"{run_short} & {fusion} & {two_stage} & {test_f1} & {neg_f1} & {neu_f1} & {pos_f1} & {val_f1} \\\\\n"
    
    latex += "\\hline\n\\end{tabular}\n\\end{table}"
    
    output_path.write_text(latex, encoding='utf-8')
    print(f"LaTeX table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Advanced analysis of run_summary.csv")
    parser.add_argument("--outputs", type=str, default="outputs", help="Outputs directory")
    parser.add_argument("--csv", type=str, help="Path to run_summary.csv (auto-detect if not specified)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top runs to display")
    parser.add_argument("--enrich", action="store_true", help="Add per-class F1 scores from eval/ folders")
    parser.add_argument("--group-by", type=str, help="Group statistics by hyperparameter (e.g., fusion, lr_clip)")
    parser.add_argument("--export", type=str, help="Export enhanced CSV to this path")
    parser.add_argument("--latex", type=str, help="Generate LaTeX table at this path")
    parser.add_argument("--metric", type=str, default="internal_test_macro_f1", 
                        help="Primary metric for ranking (default: internal_test_macro_f1)")
    
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs).expanduser().resolve()
    
    # Locate summary CSV
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = outputs_dir / "run_summary.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run cleanup.py first or specify --csv")
        return
    
    # Load summary
    print(f"Loading summary from: {csv_path}")
    df = load_summary_csv(csv_path)
    print(f"Loaded {len(df)} runs\n")
    
    # Enrich with per-class metrics
    if args.enrich:
        print("Enriching with per-class F1 scores...")
        df = enrich_with_per_class_metrics(df, outputs_dir)
    
    # Print top K runs
    print_top_k_runs(df, args.metric, args.top_k)
    
    # Per-class analysis
    if args.enrich or "negative_f1" in df.columns:
        print_per_class_analysis(df)
    
    # Grouped statistics
    if args.group_by:
        print_grouped_statistics(df, args.group_by)
    
    # Correlation analysis
    print_correlation_analysis(df)
    
    # Export enhanced CSV
    if args.export:
        export_path = Path(args.export)
        export_enhanced_csv(df, export_path)
    
    # Generate LaTeX table
    if args.latex:
        latex_path = Path(args.latex)
        generate_latex_table(df, latex_path, args.top_k)


if __name__ == "__main__":
    main()