"""
Parameter optimization command for PyneCore.
Runs a strategy with all combinations of specified parameter values (grid search)
and ranks results by a chosen metric.
"""

import csv
import gc
import json
import os
import sys
from dataclasses import dataclass, fields as dataclass_fields
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

from typer import Option, Argument, secho, Exit
from rich.progress import (Progress, SpinnerColumn, TextColumn, BarColumn,
                           MofNCompleteColumn, ProgressColumn, Task)
from rich.table import Table
from rich.text import Text
from rich.console import Console

from ..app import app, app_state
from ...core.ohlcv_file import OHLCVReader
from ...core.syminfo import SymInfo
from ...core.script_runner import ScriptRunner
from ...core.strategy_stats import StrategyStatistics

__all__ = []

console = Console()

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

METRIC_ALIASES: dict[str, str] = {
    "net_profit": "net_profit",
    "net_profit_pct": "net_profit_percent",
    "sharpe": "sharpe_ratio",
    "sortino": "sortino_ratio",
    "profit_factor": "profit_factor",
    "max_drawdown_pct": "max_equity_drawdown_percent",
    "win_rate": "percent_profitable",
    "total_trades": "total_trades",
    "avg_trade_pct": "avg_trade_percent",
    "avg_win_loss": "ratio_avg_win_loss",
}

# Metrics where lower is better
MINIMIZE_METRICS = {"max_drawdown_pct"}


# ---------------------------------------------------------------------------
# Parameter parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamSpec:
    """Immutable specification for a single parameter's sweep values."""
    name: str
    values: tuple[Any, ...]


def parse_param_specs(params_dict: dict[str, Any]) -> list[ParamSpec]:
    """Parse parameter JSON into ParamSpec list.

    Supports:
    - Range dict: {"min": 5, "max": 20, "step": 1}
    - Explicit list: [true, false] or ["ema", "sma"]
    """
    specs: list[ParamSpec] = []
    for name, spec in params_dict.items():
        if isinstance(spec, list):
            if len(spec) == 0:
                raise ValueError(f"Parameter '{name}': empty list")
            specs.append(ParamSpec(name=name, values=tuple(spec)))
        elif isinstance(spec, dict):
            required = {"min", "max", "step"}
            missing = required - set(spec.keys())
            if missing:
                raise ValueError(f"Parameter '{name}': missing keys {missing}")

            min_val, max_val, step = spec["min"], spec["max"], spec["step"]

            if step <= 0:
                raise ValueError(f"Parameter '{name}': step must be positive")
            if min_val > max_val:
                raise ValueError(f"Parameter '{name}': min ({min_val}) > max ({max_val})")

            # Integer range — use native range for precision
            if (isinstance(min_val, int) and isinstance(max_val, int)
                    and isinstance(step, int)):
                values = tuple(range(min_val, max_val + 1, step))
            else:
                # Float range — use index-based generation to avoid drift
                steps_count = int((max_val - min_val) / step) + 1
                values = []
                for i in range(steps_count):
                    val = round(min_val + i * step, 10)
                    if val > max_val + 1e-9:
                        break
                    values.append(val)
                values = tuple(values)

            if len(values) == 0:
                raise ValueError(f"Parameter '{name}': range produces no values")
            specs.append(ParamSpec(name=name, values=values))
        else:
            raise ValueError(
                f"Parameter '{name}': must be a list or dict with min/max/step, "
                f"got {type(spec).__name__}"
            )
    return specs


def generate_combinations(specs: list[ParamSpec]) -> list[dict[str, Any]]:
    """Generate all parameter combinations (cartesian product)."""
    if not specs:
        return [{}]
    names = [s.name for s in specs]
    value_lists = [s.values for s in specs]
    return [dict(zip(names, combo)) for combo in product(*value_lists)]


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------

def run_single_backtest(
    script_path: Path,
    reader: OHLCVReader,
    syminfo: SymInfo,
    start_ts: int,
    end_ts: int,
    size: int,
    params: dict[str, Any],
) -> StrategyStatistics | None:
    """Run one backtest with the given parameter overrides.

    Returns StrategyStatistics on success, None on error.
    """
    from pynecore.core import script as script_module
    from pynecore.core import function_isolation

    # 1. Pre-populate input values
    script_module._old_input_values.clear()
    for name, value in params.items():
        script_module._old_input_values[name] = value
        script_module._old_input_values[name + "__global__"] = value

    # 2. Force fresh module import
    module_name = script_path.stem
    if module_name in sys.modules:
        del sys.modules[module_name]

    # 3. Clear registered libraries from previous run
    script_module._registered_libraries.clear()

    # 4. Reset function isolation state
    function_isolation.reset()

    # 5. Fresh OHLCV iterator
    ohlcv_iter = reader.read_from(start_ts, end_ts)

    # 6. Run (no file I/O)
    try:
        runner = ScriptRunner(
            script_path, ohlcv_iter, syminfo,
            last_bar_index=size - 1,
            plot_path=None,
            strat_path=None,
            trade_path=None,
        )
        runner.run()
        return runner.stats
    except Exception as e:
        console.print(f"[yellow]Warning: Run failed for {params}: {e}[/yellow]")
        return None


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def get_metric_value(stats: StrategyStatistics, metric_attr: str) -> float:
    """Extract a metric value from StrategyStatistics."""
    return float(getattr(stats, metric_attr, 0.0))


def write_results_csv(
    results: list[tuple[dict[str, Any], StrategyStatistics]],
    param_names: list[str],
    output_path: Path,
) -> None:
    """Write all optimization results to CSV."""
    if not results:
        return

    stat_fields = [f.name for f in dataclass_fields(StrategyStatistics)]
    header = param_names + stat_fields

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for params, stats in results:
            row = [params.get(name, "") for name in param_names]
            row += [getattr(stats, field) for field in stat_fields]
            writer.writerow(row)


def write_best_toml(best_params: dict[str, Any], output_path: Path) -> None:
    """Write the best parameter set as a TOML snippet."""
    lines = [
        "# Best optimization parameters",
        "# Generated by: pyne optimize",
        "",
        "[script]",
        "",
        "# Input Settings",
        "",
    ]
    for name, value in best_params.items():
        lines.append(f"[inputs.{name}]")
        if isinstance(value, bool):
            lines.append(f"value = {str(value).lower()}")
        elif isinstance(value, str):
            lines.append(f'value = "{value}"')
        else:
            lines.append(f"value = {value}")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Progress column
# ---------------------------------------------------------------------------

class RateColumn(ProgressColumn):
    """Show runs per second."""

    def render(self, task: Task) -> Text:
        elapsed = task.elapsed
        if elapsed is None or elapsed == 0 or task.completed == 0:
            return Text("-- runs/s", style="cyan")
        rate = task.completed / elapsed
        return Text(f"{rate:.1f} runs/s", style="cyan")


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

@app.command()
def optimize(
    script: Path = Argument(
        ..., dir_okay=False, file_okay=True,
        help="Strategy script to optimize (.py)",
    ),
    data: Path = Argument(
        ..., dir_okay=False, file_okay=True,
        help="Data file to use (*.ohlcv)",
    ),
    params: Path = Argument(
        ..., dir_okay=False, file_okay=True,
        help="Parameter specification JSON file",
    ),
    metric: str = Option(
        "net_profit", "--metric", "-m",
        help="Metric to optimize. Options: "
             + ", ".join(METRIC_ALIASES.keys()),
    ),
    top_n: int = Option(10, "--top", "-n", help="Number of top results to display"),
    output: Path | None = Option(
        None, "--output", "-o",
        help="CSV output path for all results",
    ),
    save_best: bool = Option(False, "--save-best", help="Save best params as .toml"),
    time_from: datetime | None = Option(
        None, "--from", "-f",
        formats=["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"],
        help="Start date (UTC)",
    ),
    time_to: datetime | None = Option(
        None, "--to", "-t",
        formats=["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"],
        help="End date (UTC)",
    ),
):
    """
    Optimize strategy parameters via grid search.

    Runs the strategy with every combination of parameters defined in the JSON
    file and ranks results by the chosen metric.

    \b
    Example JSON (optimize.json):
    {
        "fast_length": {"min": 5, "max": 20, "step": 1},
        "slow_length": {"min": 20, "max": 50, "step": 5},
        "use_filter": [true, false]
    }
    """

    # --- Resolve script path ---
    if script.suffix != ".py":
        script = script.with_suffix(".py")
    if len(script.parts) == 1:
        script = app_state.scripts_dir / script
    if not script.exists():
        secho(f"Script file '{script}' not found!", fg="red", err=True)
        raise Exit(1)

    # --- Resolve data path ---
    if data.suffix == "":
        data = data.with_suffix(".ohlcv")
    if len(data.parts) == 1:
        data = app_state.data_dir / data
    if not data.exists():
        secho(f"Data file '{data}' not found!", fg="red", err=True)
        raise Exit(1)

    # --- Resolve params path ---
    if len(params.parts) == 1:
        # Try scripts dir first, then workdir
        candidate = app_state.scripts_dir / params
        if candidate.exists():
            params = candidate
        else:
            candidate = app_state.workdir / params
            if candidate.exists():
                params = candidate
    if not params.exists():
        secho(f"Parameter file '{params}' not found!", fg="red", err=True)
        raise Exit(1)

    # --- Validate metric ---
    if metric not in METRIC_ALIASES:
        secho(
            f"Unknown metric '{metric}'. Available: {', '.join(METRIC_ALIASES.keys())}",
            fg="red", err=True,
        )
        raise Exit(1)
    metric_attr = METRIC_ALIASES[metric]

    # --- Load parameter specs ---
    try:
        with open(params, "r") as f:
            params_dict = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        secho(f"Failed to read parameter file: {e}", fg="red", err=True)
        raise Exit(1)

    try:
        specs = parse_param_specs(params_dict)
    except ValueError as e:
        secho(f"Invalid parameter specification: {e}", fg="red", err=True)
        raise Exit(1)

    combinations = generate_combinations(specs)
    param_names = [s.name for s in specs]
    total_combos = len(combinations)

    # --- Print summary ---
    secho(f"\nOptimizing: {script.name}", fg="cyan")
    secho(f"Data:       {data.name}", fg="cyan")
    secho(f"Metric:     {metric} ({metric_attr})", fg="cyan")
    secho(f"Parameters: {len(specs)}", fg="cyan")
    for spec in specs:
        secho(f"  {spec.name}: {len(spec.values)} values "
              f"({spec.values[0]} .. {spec.values[-1]})", fg="cyan")
    secho(f"Total combinations: {total_combos}\n", fg="cyan")

    # --- Load symbol info ---
    try:
        syminfo = SymInfo.load_toml(data.with_suffix(".toml"))
    except FileNotFoundError:
        secho(f"Symbol info file '{data.with_suffix('.toml')}' not found!",
              fg="red", err=True)
        raise Exit(1)

    # --- Set optimize-mode environment ---
    saved_env: dict[str, str | None] = {}
    for key in ("PYNE_OPTIMIZE_MODE", "PYNE_SAVE_SCRIPT_TOML"):
        saved_env[key] = os.environ.get(key)
    os.environ["PYNE_OPTIMIZE_MODE"] = "1"
    os.environ["PYNE_SAVE_SCRIPT_TOML"] = "0"

    # --- Add lib directory to path ---
    lib_dir = app_state.scripts_dir / "lib"
    lib_path_added = False
    if lib_dir.exists() and lib_dir.is_dir():
        sys.path.insert(0, str(lib_dir))
        lib_path_added = True

    # --- Run optimization ---
    results: list[tuple[dict[str, Any], StrategyStatistics]] = []
    failed_count = 0

    try:
        with OHLCVReader(data) as reader:
            if not time_from:
                start_ts = reader.start_timestamp
            else:
                start_ts = int(time_from.replace(tzinfo=None).timestamp())
            if not time_to:
                end_ts = reader.end_timestamp
            else:
                end_ts = int(time_to.replace(tzinfo=None).timestamp())

            size = reader.get_size(start_ts, end_ts)

            with Progress(
                SpinnerColumn(finished_text="[green]OK"),
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[cyan]{task.percentage:>3.0f}%"),
                RateColumn(),
            ) as progress:
                task = progress.add_task("Optimizing...", total=total_combos)

                for combo in combinations:
                    stats = run_single_backtest(
                        script_path=script,
                        reader=reader,
                        syminfo=syminfo,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        size=size,
                        params=combo,
                    )

                    if stats is not None:
                        results.append((combo, stats))
                    else:
                        failed_count += 1

                    progress.update(task, advance=1)

                    # Periodic GC to prevent memory buildup
                    if len(results) % 50 == 0:
                        gc.collect()

    finally:
        # Restore environment
        for key, val in saved_env.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

        if lib_path_added and str(lib_dir) in sys.path:
            sys.path.remove(str(lib_dir))

    # --- Check results ---
    if not results:
        secho("\nNo successful runs. Cannot produce results.", fg="red", err=True)
        raise Exit(1)

    # --- Sort results ---
    is_minimize = metric in MINIMIZE_METRICS
    results.sort(
        key=lambda r: get_metric_value(r[1], metric_attr),
        reverse=not is_minimize,
    )

    # --- Display top N table ---
    display_count = min(top_n, len(results))
    table = Table(
        title=f"\nTop {display_count} Results (by {metric})",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="dim", width=4)
    for name in param_names:
        table.add_column(name, style="cyan")
    table.add_column(metric, justify="right", style="green bold")
    table.add_column("Net P %", justify="right", style="green")
    table.add_column("Sharpe", justify="right")
    table.add_column("PF", justify="right")
    table.add_column("Win %", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Max DD %", justify="right", style="red")

    for rank, (combo, stats) in enumerate(results[:display_count], 1):
        row = [str(rank)]
        row += [str(combo.get(name, "")) for name in param_names]
        row.append(f"{get_metric_value(stats, metric_attr):.4f}")
        row.append(f"{stats.net_profit_percent:.2f}")
        row.append(f"{stats.sharpe_ratio:.3f}")
        row.append(f"{stats.profit_factor:.3f}")
        row.append(f"{stats.percent_profitable:.1f}")
        row.append(str(stats.total_trades))
        row.append(f"{stats.max_equity_drawdown_percent:.2f}")
        table.add_row(*row)

    console.print(table)

    if failed_count > 0:
        console.print(f"\n[yellow]{failed_count} combination(s) failed.[/yellow]")

    console.print(
        f"\n[dim]Total: {len(results)} successful / {total_combos} combinations[/dim]"
    )

    # --- Write CSV output ---
    csv_path = output or (app_state.output_dir / f"{script.stem}_optimize.csv")
    write_results_csv(results, param_names, csv_path)
    console.print(f"[green]Results saved to: {csv_path}[/green]")

    # --- Write best TOML ---
    if save_best and results:
        best_params = results[0][0]
        toml_path = script.with_name(script.stem + ".optimized.toml")
        write_best_toml(best_params, toml_path)
        console.print(f"[green]Best parameters saved to: {toml_path}[/green]")
