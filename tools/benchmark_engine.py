import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import threading

from src.solver import Solver
from tools.testing_engine import TestingEngine, TestResult


class BenchmarkEngine:
    def __init__(
            self,
            solver: Solver,
            execution_configs: list[tuple[int, int]],
            plot_name: str,
            print_logs: bool = True,
            plot_dir: str = "plots",
            iter_save: bool = False,
            single_run_timeout: int = 120
    ):
        """
        solver_class: class implementing .solve() method
        n_values: list of integers (problem sizes)
        plot_dir: directory to save plots
        execution_configs: list of execution configurations (n, runs)
        """
        self.execution_configs = execution_configs
        self.solver = solver
        self.plot_name = plot_name
        self.results: list[TestResult] = []
        self.print_logs = print_logs
        self.plot_dir = plot_dir
        self.iter_save = iter_save
        self.single_run_timeout = single_run_timeout

    def run(self):
        test_engine = TestingEngine(self.solver, self.print_logs, self.single_run_timeout)
        for n, runs in self.execution_configs:
            test_result = test_engine.test_solver(n=n, runs=runs)
            if test_result.timeout:
                break
            self.results.append(test_result)
            self.results[-1].n = n
            if self.iter_save:
                self.save_plots()
        self.save_plots()

    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        return thread

    def generate_md_table(self, df):
        """
        Generate a Markdown table from a DataFrame with columns:
        - Header: N values
        - Rows: Time, Memory, Correct Rate
        """
        file_title = f"# Benchmark Results for {self.solver.name}"
        table_image = f"![{self.plot_name}](./{self.plot_name}.png)\n"

        header = "|  N |" + "|".join(str(n) for n in df['N']) + "|"
        separator = "|---|" + "|".join("---" for _ in df['N']) + "|"
        time_row = "|Time|" + "|".join(f"{t:.4f}" for t in df['Time']) + "|"
        memory_row = "|Memory|" + "|".join(f"{m:.2f}" for m in df['Memory']) + "|"
        correct_row = "|Correct Rate|" + "|".join(f"{c:.2f}" for c in df['Correct']) + "|"
        return "\n".join([file_title, table_image, header, separator, time_row, memory_row, correct_row])

    def save_plots(self):
        df = pd.DataFrame([r.__dict__ for r in self.results])
        df = df.rename(columns={'n': 'N', 'time': 'Time', 'peak_memory': 'Memory', 'correct': 'Correct'})

        # Use a dark background style
        plt.style.use('dark_background')

        fig, axes = plt.subplots(3, 1, figsize=(10, 15), facecolor='#222222')
        fig.patch.set_facecolor('#222222')

        # Time plot
        sns.lineplot(data=df, x='N', y='Time', marker='o', ax=axes[0], color='#1f77b4')
        axes[0].set_title('N-Queens Solver Time Usage', color='white')
        axes[0].set_ylabel('Time (seconds)', color='white')
        axes[0].set_xlabel('N', color='white')
        axes[0].tick_params(colors='white')

        # Memory plot
        sns.lineplot(data=df, x='N', y='Memory', marker='o', color='#ff7f0e', ax=axes[1])
        axes[1].set_title('N-Queens Solver Memory Usage', color='white')
        axes[1].set_ylabel('Memory (KB)', color='white')
        axes[1].set_xlabel('N', color='white')
        axes[1].tick_params(colors='white')

        # Correct rate plot
        sns.lineplot(data=df, x='N', y='Correct', marker='o', color='#2ca02c', ax=axes[2])
        axes[2].set_title('N-Queens Solver Correct Rate', color='white')
        axes[2].set_ylabel('Correct Rate(%)', color='white')
        axes[2].set_xlabel('N', color='white')
        axes[2].tick_params(colors='white')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, self.plot_name + ".png"), facecolor=fig.get_facecolor())
        plt.close()

        md_table = self.generate_md_table(df)

        md_path = os.path.join(self.plot_dir, self.plot_name + ".md")
        with open(md_path, "w") as f:
            f.write(md_table + "\n")


def aggregate_benchmarks(plot_dir: str = "plots", output_name: str = "aggregate_results"):
    """
    Aggregates all .md files in plot_dir, combines their tables, and generates a single plot
    comparing all algorithms.
    """
    import glob

    md_files = [f for f in glob.glob(os.path.join(plot_dir, "*.md")) if not f.endswith(f"{output_name}.md")]
    data = []
    algos = []
    for md_file in md_files:
        with open(md_file, "r") as f:
            lines = f.readlines()
        # Extract algorithm name from title
        algo = lines[0].replace("# Benchmark Results for ", "").strip()
        algos.append(algo)
        # Find table header
        for i, line in enumerate(lines):
            if line.startswith("|  N |"):
                table_start = i
                break
        n_values = [int(x) for x in lines[table_start].split("|")[2:-1]]
        time_row = [float(x) for x in lines[table_start + 2].split("|")[2:-1]]
        memory_row = [float(x) for x in lines[table_start + 3].split("|")[2:-1]]
        correct_row = [float(x) for x in lines[table_start + 4].split("|")[2:-1]]
        for n, t, m, c in zip(n_values, time_row, memory_row, correct_row):
            data.append({"Algorithm": algo, "N": n, "Time": t, "Memory": m, "Correct": c})

    df = pd.DataFrame(data)
    df = df.sort_values(["N", "Algorithm"])

    # Save combined markdown table
    pivot = df.pivot(index="N", columns="Algorithm", values=["Time", "Memory", "Correct"])
    md_lines = [f"# Aggregated Benchmark Results", ""]
    for metric in ["Time", "Memory", "Correct"]:
        md_lines.append(f"## {metric}")
        header = "|N|" + "|".join(pivot[metric].columns) + "|"
        sep = "|---|" + "|".join("---" for _ in pivot[metric].columns) + "|"
        md_lines.append(header)
        md_lines.append(sep)
        for n in pivot.index:
            row = f"|{n}|" + "|".join(
                f"{pivot[metric].loc[n, algo]:.4f}" if pd.notna(pivot[metric].loc[n, algo]) else "" for algo in
                pivot[metric].columns) + "|"
            md_lines.append(row)
        md_lines.append("")

    # Save markdown file
    with open(os.path.join(plot_dir, output_name + ".md"), "w") as f:
        f.write("\n".join(md_lines))

    # Plot all algorithms on same plots
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), facecolor='#222222')
    fig.patch.set_facecolor('#222222')
    colors = sns.color_palette("tab10", n_colors=len(algos))
    for i, metric in enumerate(["Time", "Memory", "Correct"]):
        for j, algo in enumerate(algos):
            subdf = df[df["Algorithm"] == algo]
            axes[i].plot(subdf["N"], subdf[metric], marker='o', label=algo, color=colors[j])
        axes[i].set_title(f"N-Queens Solver {metric} Comparison", color='white')
        ylabel = {
            "Time": "Time (seconds)",
            "Memory": "Memory (KB)",
            "Correct": "Correct Rate (%)"
        }[metric]
        axes[i].set_ylabel(ylabel, color='white')
        axes[i].set_xlabel('N', color='white')
        axes[i].tick_params(colors='white')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, output_name + ".png"), facecolor=fig.get_facecolor())
    plt.close()
