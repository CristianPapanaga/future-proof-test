"""Plot and figure production for the project.

This module produces all plots, figures and charts and saves them to
``output/figures/``. It depends only on the shared data loader
(``data_utilities``) and reuses the figure directory defined in
``log_writer``. Plotting is done with matplotlib (no display required, so
figures can be generated headlessly).
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

import data_utilities
import log_writer

#: Pre-launch (research-phase) numeric columns, shown as rows in the
#: cross-phase scatter grid. Mirrors the phase split used in ``exploration.py``.
RESEARCH_PHASE_COLUMNS: list[str] = [
    "Sample_Size",
    "Turnaround_Days",
    "Research_Cost_EUR",
    "Stated_Appeal",
    "Purchase_Intent",
    "Behavioural_Choice_Pct",
    "Implicit_Score",
]

#: Post-launch (launch-phase) numeric columns, shown as columns in the
#: cross-phase scatter grid.
LAUNCH_PHASE_COLUMNS: list[str] = [
    "Launch_Support_EUR",
    "Distribution_Pct",
    "Sales_vs_Target_Pct",
    "Repeat_Purchase_Pct",
]

#: Diverging colormap for correlation annotations: negative ``r`` → blue,
#: positive ``r`` → red, with ``r`` = 0 neutral, normalised over [-1, 1].
_CORRELATION_CMAP = plt.get_cmap("coolwarm")
_CORRELATION_NORM = Normalize(vmin=-1.0, vmax=1.0)


def _correlation_colour(r: float):
    """Map a correlation coefficient to a blue (negative) / red (positive) colour."""
    return _CORRELATION_CMAP(_CORRELATION_NORM(r))


def _every_other_tick(value: float, pos: int) -> str:
    """Tick formatter that blanks every second label to avoid overlap.

    Even-indexed ticks are blanked so that the first visible (lowest non-zero)
    tick is always labelled; the skipped values are readily inferred from the
    neighbouring ticks.
    """
    if isinstance(pos, int) and pos % 2 == 0:
        return ""
    return f"{int(value)}"


def _numeric_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce the given columns to numeric, treating ``"NA"`` markers as NaN.

    Structural ``"NA"`` (absent by design) values and ``pd.NA`` (truly missing)
    are both converted to NaN so that scatter plots and correlations use only
    the applicable, collected observations.
    """
    return pd.DataFrame(
        {
            column: pd.to_numeric(df[column], errors="coerce").astype("float64")
            for column in columns
        }
    )


def _scatter_cell(ax, xs: pd.Series, ys: pd.Series) -> None:
    """Draw a single scatter cell with its ``r`` and ``n`` annotations."""
    ax.scatter(xs, ys, s=14, alpha=0.35, linewidths=0, color="#2c7fb8")
    r = xs.corr(ys)
    ax.text(
        0.05,
        0.97,
        f"r = {r:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=_correlation_colour(r),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
    )
    ax.text(
        0.05,
        0.87,
        f"n = {len(xs)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#555555",
    )


def plot_cross_phase_scatter(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    filename: str = "cross_phase_scatter.png",
) -> Path:
    """Plot the research-phase vs launch-phase variable pairs as a scatter grid.

    A 7 x 4 grid of scatter plots is produced: research-phase variables on the
    rows, launch-phase variables on the columns. Each cell shows the
    relationship between one research metric and one launch metric, annotated
    with its Pearson correlation ``r`` and the number of observations. This
    focused grid is used instead of ``pandas.plotting.scatter_matrix`` because
    the latter would render a dense 11 x 11 matrix (121 cells) mixing in the
    within-phase pairs that are not the point of interest; the 7 x 4 grid shows
    exactly the 28 cross-phase pairs at a readable size.
    """
    output_dir = Path(output_dir) if output_dir is not None else log_writer.FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric = _numeric_matrix(df, RESEARCH_PHASE_COLUMNS + LAUNCH_PHASE_COLUMNS)

    n_rows = len(RESEARCH_PHASE_COLUMNS)
    n_cols = len(LAUNCH_PHASE_COLUMNS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 3.6, n_rows * 2.9), squeeze=False
    )

    for i, research_var in enumerate(RESEARCH_PHASE_COLUMNS):
        for j, launch_var in enumerate(LAUNCH_PHASE_COLUMNS):
            ax = axes[i, j]
            x = numeric[launch_var]
            y = numeric[research_var]
            mask = x.notna() & y.notna()
            xs, ys = x[mask], y[mask]

            _scatter_cell(ax, xs, ys)

            if j == 0:
                ax.set_ylabel(research_var, fontsize=10)
            else:
                ax.tick_params(axis="y", labelleft=False)
            if i == n_rows - 1:
                ax.set_xlabel(launch_var, fontsize=10)
            else:
                ax.tick_params(axis="x", labelbottom=False)

            if launch_var in data_utilities.CURRENCY_COLUMNS:
                ax.xaxis.set_major_formatter(FuncFormatter(_every_other_tick))

    fig.suptitle(
        "Research-phase vs launch-phase correlations", fontsize=14, y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_notable_cross_phase_scatter(
    df: pd.DataFrame,
    threshold: float = 0.2,
    output_dir: str | Path | None = None,
    filename: str = "cross_phase_scatter_notable.png",
) -> Path:
    """Plot only the research-phase x launch-phase pairs with |r| >= threshold.

    Unlike :func:`plot_cross_phase_scatter`, which shows all 28 cross-phase
    pairs in a fixed 7 x 4 grid, this plots only the pairs whose Pearson
    correlation reaches ``threshold`` (default 0.2, i.e. at least a "weak"
    correlation), sorted by |r| and laid out in a compact grid.
    """
    output_dir = Path(output_dir) if output_dir is not None else log_writer.FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric = _numeric_matrix(df, RESEARCH_PHASE_COLUMNS + LAUNCH_PHASE_COLUMNS)

    pairs = []
    for research_var in RESEARCH_PHASE_COLUMNS:
        for launch_var in LAUNCH_PHASE_COLUMNS:
            x = numeric[launch_var]
            y = numeric[research_var]
            mask = x.notna() & y.notna()
            xs, ys = x[mask], y[mask]
            r = xs.corr(ys)
            if abs(r) >= threshold:
                pairs.append((research_var, launch_var, xs, ys, r))
    pairs.sort(key=lambda item: abs(item[4]), reverse=True)

    n = len(pairs)
    n_cols = 2
    n_rows = ceil(n / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.0), squeeze=False
    )

    for idx, (research_var, launch_var, xs, ys, r) in enumerate(pairs):
        i, j = divmod(idx, n_cols)
        ax = axes[i, j]
        _scatter_cell(ax, xs, ys)
        ax.set_xlabel(launch_var, fontsize=10)
        ax.set_ylabel(research_var, fontsize=10)

        if launch_var in data_utilities.CURRENCY_COLUMNS:
            ax.xaxis.set_major_formatter(FuncFormatter(_every_other_tick))

    for idx in range(n, n_rows * n_cols):
        i, j = divmod(idx, n_cols)
        axes[i, j].axis("off")

    fig.suptitle(
        f"Notable research-phase vs launch-phase correlations (|r| ≥ {threshold:.2f})\n"
        "Colour: negative r (blue) → positive r (red)",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    data = data_utilities.load_historical_data()
    full_path = plot_cross_phase_scatter(data)
    notable_path = plot_notable_cross_phase_scatter(data)
    print(f"Full cross-phase grid saved to: {full_path}")
    print(f"Notable cross-phase grid saved to: {notable_path}")

