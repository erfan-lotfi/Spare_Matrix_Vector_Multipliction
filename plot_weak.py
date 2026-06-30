#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    numeric_cols = ["rows", "cols", "nnz", "workers", "total_time", 
                    "compute_time", "comm_time", "gflops", "speedup", "max_error"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["workers", "total_time", "rows", "cols", "nnz"])
    df = df[df["workers"] > 0]
    df = df[df["rows"] > 0]
    return df

def compute_density(df):
    df = df.copy()
    df["density"] = df["nnz"] / (df["rows"] * df["cols"])
    return df

def save_plot(output_dir, name):
    out = os.path.join(output_dir, name)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"Saved: {out}")
    plt.close()

def plot_weak_scaling(df, output_dir, title_prefix=""):
    df = df[df["mode"].isin(["openmp", "mpi"])].copy()
    if df.empty:
        print("No OpenMP or MPI data found!")
        return
    
    df = compute_density(df)
    any_plot = False
    
    for mode, grp in df.groupby("mode"):
        grp = grp.sort_values("workers")
        if len(grp) < 2:
            print(f"Insufficient data for {mode}")
            continue
        
        density_val = grp["density"].iloc[0]
        plt.figure(figsize=(10, 6))
        
        color = "blue" if mode == "openmp" else "red"
        plt.plot(grp["workers"], grp["total_time"], 
                marker="o", markersize=10, linestyle="-", linewidth=3,
                color=color, label=f"{mode.upper()}")
        
        for _, row in grp.iterrows():
            label = f"{int(row['rows'])}x{int(row['cols'])}\nnnz={int(row['nnz'])}"
            plt.annotate(label, (row["workers"], row["total_time"]),
                        textcoords="offset points", xytext=(10, 10),
                        fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        ideal_time = grp["total_time"].iloc[0]
        plt.axhline(y=ideal_time, color="green", linestyle="--", 
                   linewidth=2, alpha=0.7, label=f"Ideal (constant time = {ideal_time:.6f}s)")
        
        title = f"Weak Scaling - {mode.upper()}\n{title_prefix}density={density_val:.6f}"
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Workers (threads/processes)", fontsize=12)
        plt.ylabel("Total Time (s)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", fontsize=11)
        plt.xticks(grp["workers"].unique())
        
        filename = f"weak_scaling_{mode}_density_{density_val:.6f}.png"
        save_plot(output_dir, filename)
        any_plot = True
    
    if not any_plot:
        print("No Weak Scaling plots were generated!")

def plot_weak_scaling_combined(df, output_dir, title_prefix=""):
    df = df[df["mode"].isin(["openmp", "mpi"])].copy()
    if df.empty:
        print("No OpenMP or MPI data found!")
        return
    
    df = compute_density(df)
    density_val = df["density"].iloc[0] if not df.empty else 0
    plt.figure(figsize=(12, 7))
    
    colors = {"openmp": "blue", "mpi": "red"}
    markers = {"openmp": "o", "mpi": "s"}
    
    for mode, grp in df.groupby("mode"):
        grp = grp.sort_values("workers")
        plt.plot(grp["workers"], grp["total_time"], 
                marker=markers.get(mode, "o"), markersize=12,
                linestyle="-", linewidth=3, color=colors.get(mode, "black"),
                label=f"{mode.upper()}")
        
        for _, row in grp.iterrows():
            label = f"{int(row['rows'])}x{int(row['cols'])}"
            plt.annotate(label, (row["workers"], row["total_time"]),
                        textcoords="offset points", xytext=(10, 5),
                        fontsize=8, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))
    
    if not df.empty:
        first_time = df[df["workers"] == df["workers"].min()]["total_time"].iloc[0]
        plt.axhline(y=first_time, color="green", linestyle="--", 
                   linewidth=2, alpha=0.7, label=f"Ideal (constant time ≈ {first_time:.6f}s)")
    
    title = f"Weak Scaling - Combined\n{title_prefix}density ≈ {density_val:.6f}"
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Workers (threads/processes)", fontsize=12)
    plt.ylabel("Total Time (s)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=12)
    plt.xticks(sorted(df["workers"].unique()))
    
    save_plot(output_dir, "weak_scaling_combined.png")

def main():
    parser = argparse.ArgumentParser(description="Plot Weak Scaling graphs for SpMV project")
    parser.add_argument("csv_file", help="CSV file containing Weak Scaling data")
    parser.add_argument("--output-dir", default="plots_weak", help="Output directory for graphs")
    parser.add_argument("--title", default="", help="Title prefix for graphs")
    parser.add_argument("--combined", action="store_true", help="Plot combined graph (OpenMP + MPI)")
    
    args = parser.parse_args()
    ensure_output_dir(args.output_dir)
    
    df = load_data(args.csv_file)
    if df.empty:
        print("CSV file is empty or contains no valid data!")
        return
    
    print("\n" + "="*60)
    print("Weak Scaling Data Summary")
    print("="*60)
    print(f"Total records: {len(df)}")
    print(f"Modes: {sorted(df['mode'].unique())}")
    print(f"Workers: {sorted(df['workers'].unique())}")
    print("\nMatrix dimensions:")
    for _, row in df[["rows", "cols", "nnz", "workers"]].drop_duplicates().sort_values("workers").iterrows():
        density = row["nnz"] / (row["rows"] * row["cols"])
        print(f"  workers={row['workers']}: {row['rows']}x{row['cols']}, nnz={row['nnz']}, density={density:.6f}")
    print("="*60 + "\n")
    
    if args.combined:
        plot_weak_scaling_combined(df, args.output_dir, args.title)
    else:
        plot_weak_scaling(df, args.output_dir, args.title)
    
    print(f"\nAll graphs saved to '{args.output_dir}'")

if __name__ == "__main__":
    main()