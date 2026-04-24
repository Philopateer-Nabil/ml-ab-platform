"""Main CLI — ``mlab <command>``."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ml_ab_platform.analysis import StatisticalAnalyzer
from ml_ab_platform.config import get_settings
from ml_ab_platform.experiments import (
    ExperimentCreate,
    ExperimentManager,
    ExperimentStore,
)
from ml_ab_platform.logging_ import configure_logging

console = Console()


@click.group()
@click.option("--config", "-c", type=click.Path(), default=None,
              help="Path to a YAML config file (overrides default search).")
def cli(config: str | None) -> None:
    """ML A/B Testing Platform command-line interface."""
    configure_logging()
    if config:
        import os
        os.environ["MLAB_CONFIG"] = config


# ------------------------------- train -------------------------------------- #
@cli.command()
@click.option("--data", "data_path", type=click.Path(), default=None,
              help="Path to UCI Adult CSV (synthesised if missing).")
def train(data_path: str | None) -> None:
    """Train Model A (RandomForest) and Model B (XGBoost) and save artifacts."""
    from ml_ab_platform.models.training import train_and_save_models
    baseline = train_and_save_models(data_path=data_path)
    console.print("[green]Training complete.[/green]")
    console.print_json(json.dumps(baseline, indent=2))


# ------------------------------- serve -------------------------------------- #
@cli.command()
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
def serve(host: str | None, port: int | None) -> None:
    """Launch the FastAPI gateway."""
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "ml_ab_platform.api.app:app",
        host=host or settings.api.host,
        port=port or settings.api.port,
        log_level=settings.api.log_level,
        reload=False,
    )


# ------------------------------- dashboard ---------------------------------- #
@cli.command()
@click.option("--port", default=None, type=int)
def dashboard(port: int | None) -> None:
    """Launch the Streamlit dashboard."""
    settings = get_settings()
    app_path = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(port or settings.dashboard.port),
        "--server.headless", "true",
    ]
    console.print(f"[cyan]Launching Streamlit:[/cyan] {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


# ------------------------------- experiment --------------------------------- #
@cli.group()
def experiment() -> None:
    """Experiment lifecycle commands."""


@experiment.command("create")
@click.option("--name", required=True)
@click.option("--description", default="")
@click.option("--strategy", type=click.Choice(["fixed", "sticky", "bandit", "canary"]),
              default="fixed")
@click.option("--split", type=float, default=0.5, help="Fraction to Model A (0-1).")
@click.option("--model-a", default="A")
@click.option("--model-b", default="B")
@click.option("--target-metric", default="accuracy")
@click.option("--min-samples", type=int, default=500)
def experiment_create(name: str, description: str, strategy: str, split: float,
                      model_a: str, model_b: str, target_metric: str,
                      min_samples: int) -> None:
    """Create a new experiment."""
    config = {"split": split} if strategy in ("fixed", "sticky") else {}
    if strategy == "canary":
        config = {"initial_split": min(split, 0.2), "promoted_split": 0.5}
    mgr = ExperimentManager()
    exp = mgr.create(ExperimentCreate(
        name=name, description=description,
        routing_strategy=strategy, routing_config=config,
        model_a=model_a, model_b=model_b,
        target_metric=target_metric, minimum_sample_size=min_samples,
    ))
    console.print(f"[green]Created experiment {exp.id}[/green]")


@experiment.command("start")
@click.argument("experiment_id")
def experiment_start(experiment_id: str) -> None:
    """Start routing traffic through this experiment."""
    from ml_ab_platform.experiments.manager import ExperimentAlreadyRunningError
    mgr = ExperimentManager()
    try:
        exp = mgr.start(experiment_id)
    except ExperimentAlreadyRunningError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    except KeyError:
        console.print(f"[red]No such experiment: {experiment_id}[/red]")
        sys.exit(1)
    console.print(f"[green]Started {exp.id} ({exp.name}).[/green]")


@experiment.command("stop")
@click.argument("experiment_id")
def experiment_stop(experiment_id: str) -> None:
    """Stop an experiment."""
    mgr = ExperimentManager()
    try:
        exp = mgr.stop(experiment_id)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    console.print(f"[yellow]Stopped {exp.id}.[/yellow]")


@experiment.command("conclude")
@click.argument("experiment_id")
@click.option("--winner", default=None, help="A/B or leave empty for auto.")
def experiment_conclude(experiment_id: str, winner: str | None) -> None:
    """Conclude and archive a stopped experiment."""
    mgr = ExperimentManager()
    exp = mgr.conclude(experiment_id, winner=winner)
    console.print(f"[green]Concluded {exp.id} → winner: {exp.winner}[/green]")


@experiment.command("list")
def experiment_list() -> None:
    """List all experiments."""
    store = ExperimentStore()
    t = Table(title="Experiments")
    t.add_column("ID")
    t.add_column("Name")
    t.add_column("Status")
    t.add_column("Strategy")
    t.add_column("Winner")
    t.add_column("Created")
    for exp in store.list():
        t.add_row(exp.id[:8], exp.name, exp.status.value, exp.routing_strategy,
                  exp.winner or "—", exp.created_at.isoformat(timespec="seconds"))
    console.print(t)


@experiment.command("status")
@click.argument("experiment_id")
def experiment_status(experiment_id: str) -> None:
    """Print current experiment status and headline metrics."""
    store = ExperimentStore()
    exp = store.get(experiment_id)
    if exp is None:
        console.print(f"[red]No such experiment: {experiment_id}[/red]")
        sys.exit(1)
    console.print(f"[bold]{exp.name}[/bold] — {exp.status.value}")
    console.print(f"Strategy: {exp.routing_strategy}")
    analysis = StatisticalAnalyzer().analyze(experiment_id).to_dict()
    _render_analysis(analysis)


# ------------------------------- analyze ------------------------------------ #
@cli.command()
@click.argument("experiment_id")
def analyze(experiment_id: str) -> None:
    """Run full statistical analysis and print a report."""
    analysis = StatisticalAnalyzer().analyze(experiment_id).to_dict()
    _render_analysis(analysis)


def _render_analysis(analysis: dict) -> None:
    verdict = analysis.get("verdict", "?")
    console.print(f"\n[bold]Verdict:[/bold] {verdict.upper().replace('_', ' ')}")
    console.print(f"[dim]{analysis.get('verdict_reason', '')}[/dim]\n")

    for m in (analysis.get("model_a"), analysis.get("model_b")):
        if not m:
            continue
        console.print(f"[cyan]Model {m['version']}:[/cyan] "
                      f"n_pred={m['n_predictions']}, n_fb={m['n_feedback']}, "
                      f"acc={m['accuracy']:.4f}, "
                      f"p50_lat={m['latency_p50']:.1f}ms, "
                      f"p95_lat={m['latency_p95']:.1f}ms")

    acc = analysis.get("accuracy_test")
    if acc:
        console.print(
            f"\nz-test: z={acc['z']:.3f}, p={acc['p_value']:.4f}, "
            f"delta={acc['diff']*100:.2f}pp, "
            f"95% CI=[{acc['ci_low']*100:.2f}, {acc['ci_high']*100:.2f}]pp, "
            f"Cohen's h={acc['cohen_h']:.3f}"
        )
    if analysis.get("required_sample_size"):
        console.print(
            f"Required n/arm: {analysis['required_sample_size']}, "
            f"current power: {(analysis.get('current_power') or 0):.2%}, "
            f"O'Brien-Fleming z≥{(analysis.get('obrien_fleming_critical_z') or 0):.2f} "
            f"({'crossed' if analysis.get('sequential_significant') else 'not yet'})"
        )
    for w in analysis.get("warnings", []):
        console.print(f"[yellow]⚠ {w}[/yellow]")


# ------------------------------- simulate ----------------------------------- #
@cli.command()
@click.option("--scenario", type=click.Choice(
    ["equal", "clear-winner", "subtle-winner", "degradation"]),
    default="clear-winner")
@click.option("--requests", "requests_", default=2000, type=int)
@click.option("--delay-ms", default=5, type=int)
@click.option("--feedback-delay-ms", default=50, type=int)
@click.option("--api-url", default=None)
@click.option("--seed", default=0, type=int)
def simulate(scenario: str, requests_: int, delay_ms: int, feedback_delay_ms: int,
             api_url: str | None, seed: int) -> None:
    """Run a synthetic traffic scenario against the API."""
    from ml_ab_platform.simulation import Simulator
    settings = get_settings()
    base = api_url or f"http://{settings.api.host}:{settings.api.port}"
    sim = Simulator(
        api_url=base, scenario=scenario,
        requests_per_run=requests_, delay_ms=delay_ms,
        feedback_delay_ms=feedback_delay_ms, seed=seed,
    )
    summary = sim.run()
    console.print_json(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    cli()
