import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def load_data(csv_file):
    df = pd.read_csv(csv_file)

    required_cols = [
        "mode", "format", "rows", "cols", "nnz", "workers",
        "total_time", "compute_time", "comm_time",
        "gflops", "speedup", "max_error"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    return df


def add_problem_size_label(df):
    df = df.copy()
    df["problem_label"] = (
        df["rows"].astype(str) + "x" +
        df["cols"].astype(str) + ", nnz=" +
        df["nnz"].astype(str)
    )
    return df


def filter_by_format(df, data_format=None):
    if data_format is None:
        return df
    return df[df["format"] == data_format].copy()


def save_plot(output_dir, name):
    out = os.path.join(output_dir, name)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"Saved: {out}")
    plt.close()


def plot_speedup(df, output_dir):
    """
    Speedup vs workers for each mode and each fixed problem size
    """
    df = df[df["mode"].isin(["openmp", "mpi"])].copy()
    if df.empty:
        print("No OpenMP/MPI rows found for speedup plot.")
        return

    for problem, grp in df.groupby("problem_label"):
        plt.figure(figsize=(8, 5))
        for mode, g in grp.groupby("mode"):
            g = g.sort_values("workers")
            plt.plot(g["workers"], g["speedup"], marker="o", label=mode)

        max_workers = grp["workers"].max()
        ideal_x = list(range(1, int(max_workers) + 1))
        ideal_y = ideal_x
        plt.plot(ideal_x, ideal_y, "--", label="ideal")

        plt.title(f"Speedup\n{problem}")
        plt.xlabel("Workers (threads/processes)")
        plt.ylabel("Speedup")
        plt.grid(True, alpha=0.3)
        plt.legend()

        safe_name = problem.replace(" ", "_").replace(",", "").replace("=", "-")
        save_plot(output_dir, f"speedup_{safe_name}.png")


def plot_gflops(df, output_dir):
    """
    GFLOPS vs workers for each mode and fixed problem size
    """
    df = df[df["mode"].isin(["serial", "openmp", "mpi"])].copy()
    if df.empty:
        print("No rows found for GFLOPS plot.")
        return

    for problem, grp in df.groupby("problem_label"):
        plt.figure(figsize=(8, 5))

        for mode, g in grp.groupby("mode"):
            g = g.sort_values("workers")
            plt.plot(g["workers"], g["gflops"], marker="o", label=mode)

        plt.title(f"GFLOPS\n{problem}")
        plt.xlabel("Workers (threads/processes)")
        plt.ylabel("GFLOPS")
        plt.grid(True, alpha=0.3)
        plt.legend()

        safe_name = problem.replace(" ", "_").replace(",", "").replace("=", "-")
        save_plot(output_dir, f"gflops_{safe_name}.png")


def plot_strong_scaling(df, output_dir):
    """
    Strong scaling:
    fixed rows, cols, nnz
    plot total_time vs workers
    """
    df = df[df["mode"].isin(["openmp", "mpi"])].copy()
    if df.empty:
        print("No OpenMP/MPI rows found for strong scaling plot.")
        return

    grouped = df.groupby(["rows", "cols", "nnz"])
    for (rows, cols, nnz), grp in grouped:
        plt.figure(figsize=(8, 5))

        for mode, g in grp.groupby("mode"):
            g = g.sort_values("workers")
            plt.plot(g["workers"], g["total_time"], marker="o", label=mode)

        plt.title(f"Strong Scaling\n{rows}x{cols}, nnz={nnz}")
        plt.xlabel("Workers (threads/processes)")
        plt.ylabel("Total Time (s)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        save_plot(output_dir, f"strong_scaling_{rows}x{cols}_nnz-{nnz}.png")


def compute_density(df):
    df = df.copy()
    df["density"] = df["nnz"] / (df["rows"] * df["cols"])
    return df


def plot_weak_scaling(df, output_dir, tolerance=1e-12):
    """
    Weak scaling:
    problem size changes with workers, but density remains تقریبا ثابت
    plot total_time vs workers for same mode and same density
    """
    df = df[df["mode"].isin(["openmp", "mpi"])].copy()
    if df.empty:
        print("No OpenMP/MPI rows found for weak scaling plot.")
        return

    df = compute_density(df)

    # rounded density to group close floating-point values
    df["density_round"] = df["density"].round(12)

    any_plot = False
    for (mode, density_round), grp in df.groupby(["mode", "density_round"]):
        if grp["workers"].nunique() < 2:
            continue
        if grp["problem_label"].nunique() < 2:
            continue

        grp = grp.sort_values("workers")

        plt.figure(figsize=(8, 5))
        plt.plot(grp["workers"], grp["total_time"], marker="o", label=f"{mode}")

        # annotate sizes
        for _, row in grp.iterrows():
            label = f'{int(row["rows"])}x{int(row["cols"])}, nnz={int(row["nnz"])}'
            plt.annotate(label, (row["workers"], row["total_time"]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)

        plt.title(f"Weak Scaling\nmode={mode}, density={density_round}")
        plt.xlabel("Workers (threads/processes)")
        plt.ylabel("Total Time (s)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        save_plot(output_dir, f"weak_scaling_{mode}_density-{density_round}.png")
        any_plot = True

    if not any_plot:
        print("No suitable grouped data found for weak scaling.")
        print("Check that density is constant across multiple worker counts.")


def plot_mpi_comm_time(df, output_dir):
    """
    MPI Communication Time vs workers
    """
    df = df[df["mode"] == "mpi"].copy()
    if df.empty:
        print("No MPI rows found for communication-time plot.")
        return

    for problem, grp in df.groupby("problem_label"):
        grp = grp.sort_values("workers")

        plt.figure(figsize=(8, 5))
        plt.plot(grp["workers"], grp["comm_time"], marker="o", label="MPI comm time")
        plt.plot(grp["workers"], grp["compute_time"], marker="s", label="MPI compute time")
        plt.plot(grp["workers"], grp["total_time"], marker="^", label="MPI total time")

        plt.title(f"MPI Times\n{problem}")
        plt.xlabel("Processes")
        plt.ylabel("Time (s)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        safe_name = problem.replace(" ", "_").replace(",", "").replace("=", "-")
        save_plot(output_dir, f"mpi_times_{safe_name}.png")


def plot_efficiency(df, output_dir):
    """
    Optional extra plot: efficiency = speedup / workers
    """
    df = df[df["mode"].isin(["openmp", "mpi"])].copy()
    if df.empty:
        print("No OpenMP/MPI rows found for efficiency plot.")
        return

    df["efficiency"] = df["speedup"] / df["workers"]

    for problem, grp in df.groupby("problem_label"):
        plt.figure(figsize=(8, 5))
        for mode, g in grp.groupby("mode"):
            g = g.sort_values("workers")
            plt.plot(g["workers"], g["efficiency"], marker="o", label=mode)

        plt.title(f"Parallel Efficiency\n{problem}")
        plt.xlabel("Workers")
        plt.ylabel("Efficiency")
        plt.grid(True, alpha=0.3)
        plt.legend()

        safe_name = problem.replace(" ", "_").replace(",", "").replace("=", "-")
        save_plot(output_dir, f"efficiency_{safe_name}.png")


def print_summary(df):
    print("\n===== Data Summary =====")
    print(f"Rows in CSV: {len(df)}")
    print("Modes:", sorted(df["mode"].unique()))
    print("Formats:", sorted(df["format"].unique()))
    print("Workers:", sorted(df["workers"].unique()))
    print("Problem sizes:")
    for _, row in df[["rows", "cols", "nnz"]].drop_duplicates().iterrows():
        print(f'  - {row["rows"]} x {row["cols"]}, nnz={row["nnz"]}')
    print("========================\n")


def main():
    parser = argparse.ArgumentParser(description="Plot SpMV benchmark results from CSV")
    parser.add_argument("csv_file", help="Input CSV file, e.g. results.csv")
    parser.add_argument("--output-dir", default="plots", help="Directory to save plots")
    parser.add_argument("--format", default=None, help="Filter by data format: text or bin")
    parser.add_argument("--no-efficiency", action="store_true", help="Disable efficiency plots")
    args = parser.parse_args()

    ensure_output_dir(args.output_dir)

    df = load_data(args.csv_file)
    df = add_problem_size_label(df)
    df = filter_by_format(df, args.format)

    if df.empty:
        print("No data left after filtering.")
        return

    print_summary(df)

    plot_speedup(df, args.output_dir)
    plot_gflops(df, args.output_dir)
    plot_strong_scaling(df, args.output_dir)
    plot_weak_scaling(df, args.output_dir)
    plot_mpi_comm_time(df, args.output_dir)

    if not args.no_efficiency:
        plot_efficiency(df, args.output_dir)

    print("All requested plots are generated.")


if __name__ == "__main__":
    main()

