"""Plot and figure production for the project.

This module produces all plots, figures and charts and saves them to
``output/figures/``. It depends on the shared data loader (``data_utilities``)
and, for the model-fit plots, on ``modelling``; it reuses the figure directory
defined in ``log_writer``. Plotting is done with matplotlib (no display
required, so figures can be generated headlessly).
"""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

import data_utilities
import log_writer
import modelling

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


def _model_specs() -> list[tuple[str, list[str]]]:
    """Predictor sets for the three model stages, from survey to implicit."""
    return [
        ("Survey", modelling.SURVEY_MEASURES),
        (
            "Survey + behavioural",
            modelling.SURVEY_MEASURES + modelling.BEHAVIOURAL_MEASURES,
        ),
        (
            "Survey + behavioural + implicit",
            modelling.SURVEY_MEASURES
            + modelling.BEHAVIOURAL_MEASURES
            + modelling.IMPLICIT_MEASURES,
        ),
    ]


def _draw_fitted_cell(
    ax, fit: dict, lo: float, hi: float, r2_label: str = "R²", n_label: str = "n"
) -> None:
    """Draw one observed-vs-fitted scatter cell (points + identity line)."""
    ax.scatter(
        fit["y"], fit["y_pred"], s=16, alpha=0.5, linewidths=0, color="#2c7fb8"
    )
    ax.plot([lo, hi], [lo, hi], color="#888888", linestyle="--", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.04,
        0.96,
        f"{r2_label} = {fit['r2']:.3f}\n{n_label} = {fit['n']}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
    )


def _fitted_limits(fits: list[dict]) -> tuple[float, float]:
    """Return padded (lo, hi) limits spanning all fits' observed/predicted values."""
    lo = min(min(f["y"].min(), f["y_pred"].min()) for f in fits)
    hi = max(max(f["y"].max(), f["y_pred"].max()) for f in fits)
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def _plot_fitted_grid(
    specs: list[tuple[str, list[str]]],
    fits: list[dict],
    suptitle: str,
    output_dir: Path,
    filename: str,
    r2_label: str = "R²",
    n_label: str = "n",
) -> Path:
    """Draw a 1 x 3 observed-vs-fitted grid from already-fitted models."""
    lo, hi = _fitted_limits(fits)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for ax, (title, _), fit in zip(axes, specs, fits):
        _draw_fitted_cell(ax, fit, lo, hi, r2_label=r2_label, n_label=n_label)
        ax.set_title(title, fontsize=11)

    for ax in axes:
        ax.set_xlabel("Observed Sales_vs_Target_Pct", fontsize=10)
    axes[0].set_ylabel("Predicted Sales_vs_Target_Pct", fontsize=10)

    fig.suptitle(suptitle, fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_regression_comparison(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    filename: str = "regression_comparison.png",
) -> Path:
    """Plot OLS (top row) and ridge (bottom row) fits side by side.

    A 2 x 3 grid is produced: the top row shows the three OLS models and the
    bottom row the three ridge models, all on identical axis limits so the
    spread of points relative to the identity line is directly comparable.
    """
    output_dir = Path(output_dir) if output_dir is not None else log_writer.FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = _model_specs()
    ols_fits = [modelling.linear_model(df, predictors) for _, predictors in specs]
    ridge_fits = [modelling.ridge_model(df, predictors) for _, predictors in specs]
    lo, hi = _fitted_limits(ols_fits + ridge_fits)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))

    for col, (title, _) in enumerate(specs):
        _draw_fitted_cell(axes[0, col], ols_fits[col], lo, hi)
        axes[0, col].set_title(f"OLS — {title}", fontsize=11)
        _draw_fitted_cell(axes[1, col], ridge_fits[col], lo, hi)
        axes[1, col].set_title(f"Ridge — {title}", fontsize=11)

    for col in range(3):
        axes[1, col].set_xlabel("Observed Sales_vs_Target_Pct", fontsize=10)
    for row in range(2):
        axes[row, 0].set_ylabel("Predicted Sales_vs_Target_Pct", fontsize=10)

    fig.suptitle(
        "OLS vs ridge: observed vs fitted Sales_vs_Target_Pct", fontsize=13, y=0.99
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_regression_fits(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    filename: str = "regression_fits.png",
) -> Path:
    """Plot observed vs fitted ``Sales_vs_Target_Pct`` for the three OLS models.

    One scatter panel per model, side by side, with the identity line overlaid
    and each panel annotated with its R² and sample size.
    """
    output_dir = Path(output_dir) if output_dir is not None else log_writer.FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _model_specs()
    fits = [modelling.linear_model(df, predictors) for _, predictors in specs]
    return _plot_fitted_grid(
        specs,
        fits,
        "Observed vs fitted Sales_vs_Target_Pct by model (OLS)",
        output_dir,
        filename,
    )


def plot_ridge_regression_fits(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    filename: str = "ridge_regression_fits.png",
) -> Path:
    """Plot observed vs fitted ``Sales_vs_Target_Pct`` for the three ridge models.

    Mirrors :func:`plot_regression_fits` (one side-by-side panel per model, with
    the identity line and R²/n annotations), but fits with ``modelling.
    ridge_model`` so the panels can be compared directly against the OLS figure.
    """
    output_dir = Path(output_dir) if output_dir is not None else log_writer.FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _model_specs()
    fits = [modelling.ridge_model(df, predictors) for _, predictors in specs]
    return _plot_fitted_grid(
        specs,
        fits,
        "Observed vs fitted Sales_vs_Target_Pct by model (ridge)",
        output_dir,
        filename,
    )


def plot_time_validation(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    filename: str = "time_validation.png",
) -> Path:
    """Plot observed vs predicted ``Sales_vs_Target_Pct`` on the 2025 test year.

    Each ridge model is trained on 2022–2024 and evaluated on the held-out 2025
    concepts, so the panels show genuinely out-of-sample predictions. Each cell
    is annotated with the test-year R² and the number of test observations.
    """
    output_dir = Path(output_dir) if output_dir is not None else log_writer.FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = _model_specs()
    fits = []
    for _, predictors in specs:
        result = modelling.ridge_train_test(df, predictors)
        fits.append(
            {
                "y": result["y_test"],
                "y_pred": result["y_pred"],
                "r2": result["r2_test"],
                "n": result["n_test"],
            }
        )

    return _plot_fitted_grid(
        specs,
        fits,
        "Test year (2025): observed vs predicted Sales_vs_Target_Pct (ridge)",
        output_dir,
        filename,
        r2_label="Test R²",
        n_label="n test",
    )


def plot_repeated_cv_diffs(
    df: pd.DataFrame,
    base_predictors: list[str],
    full_predictors: list[str],
    base_label: str,
    full_label: str,
    suptitle: str,
    output_dir: str | Path | None = None,
    filename: str = "repeated_cv_diffs.png",
) -> Path:
    """Histogram of fold-level (full − base) metric differences.

    Two rows (fixed-α vs per-fold RidgeCV) by three columns (R², RMSE, MAE).
    Each panel shows the distribution of within-fold differences across the
    repeated K-fold splits, with the confidence interval shaded, a dashed line
    at zero (no difference), and a solid line at the mean.
    """
    output_dir = Path(output_dir) if output_dir is not None else log_writer.FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed = modelling.repeated_kfold_compare(
        df, base_predictors, full_predictors, alpha=modelling.FIXED_ALPHA
    )
    cv = modelling.repeated_kfold_compare(
        df, base_predictors, full_predictors, alpha=None
    )

    diff_label = f"{full_label} − {base_label}"
    strategies = [
        ("Option A — fixed α = 10", fixed),
        ("Option B — per-fold RidgeCV", cv),
    ]
    metrics = [
        ("Δ R²", "r2_diff", f"{diff_label} R²", False),
        ("Δ RMSE", "rmse_diff", f"{diff_label} RMSE", True),
        ("Δ MAE", "mae_diff", f"{diff_label} MAE", True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))

    for row, (strat_title, res) in enumerate(strategies):
        for col, (metric_title, key, xlabel, lower_is_better) in enumerate(metrics):
            ax = axes[row, col]
            diffs = res[key]
            s = modelling.summarise_fold_diffs(diffs)
            ax.axvspan(s["ci_lo"], s["ci_hi"], color="red", alpha=0.15, zorder=0)
            ax.hist(diffs, bins=25, color="#2c7fb8", alpha=0.85, edgecolor="white")
            ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
            ax.axvline(s["mean"], color="black", linewidth=2)
            better = s["p_positive"] if not lower_is_better else 1 - s["p_positive"]
            ax.text(
                0.03,
                0.97,
                f"mean = {s['mean']:+.3f}\n"
                f"95% CI [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]\n"
                f"{full_label} better: {better * 100:.0f}% of folds",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
            )
            ax.set_title(f"{strat_title} — {metric_title}", fontsize=11)
            if row == 1:
                ax.set_xlabel(xlabel, fontsize=10)
            if col == 0:
                ax.set_ylabel("Folds", fontsize=10)

    fig.suptitle(suptitle, fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_classification_cv_diffs(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    filename: str = "classification_cv_diffs.png",
) -> Path:
    """Histogram of fold-level classification metric differences.

    Two rows (implicit vs behavioural, behavioural vs survey) by three columns
    (ROC-AUC, balanced accuracy, F1). Each panel shows the distribution of
    within-fold (full − base) differences across the repeated K-fold splits,
    with the confidence interval shaded, a dashed line at zero, and a solid
    line at the mean.
    """
    output_dir = Path(output_dir) if output_dir is not None else log_writer.FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    comparisons = [
        (
            modelling.BEHAVIOURAL_CV_PREDICTORS,
            modelling.IMPLICIT_CV_PREDICTORS,
            "behavioural",
            "implicit",
        ),
        (
            modelling.SURVEY_MEASURES,
            modelling.BEHAVIOURAL_CV_PREDICTORS,
            "survey",
            "behavioural",
        ),
    ]
    metrics = [
        ("Δ ROC-AUC", "auc_diff", "ROC-AUC"),
        ("Δ balanced accuracy", "bal_diff", "balanced accuracy"),
        ("Δ F1", "f1_diff", "F1"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))

    for row, (base, full, base_label, full_label) in enumerate(comparisons):
        res = modelling.repeated_kfold_compare_logistic(df, base, full)
        diff_label = f"{full_label} − {base_label}"
        for col, (metric_title, key, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            diffs = res[key]
            s = modelling.summarise_fold_diffs(diffs)
            ax.axvspan(s["ci_lo"], s["ci_hi"], color="red", alpha=0.15, zorder=0)
            ax.hist(diffs, bins=25, color="#2c7fb8", alpha=0.85, edgecolor="white")
            ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
            ax.axvline(s["mean"], color="black", linewidth=2)
            ax.text(
                0.03,
                0.97,
                f"mean = {s['mean']:+.3f}\n"
                f"95% CI [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]\n"
                f"{full_label} better: {s['p_positive'] * 100:.0f}% of folds",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
            )
            ax.set_title(f"{diff_label}: {metric_title}", fontsize=11)
            if row == 1:
                ax.set_xlabel(f"Δ {metric_label}", fontsize=10)
            if col == 0:
                ax.set_ylabel("Folds", fontsize=10)

    fig.suptitle(
        "Repeated K-fold CV (classification): fold differences (full − base)",
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_precision_recall(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    filename: str = "precision_recall.png",
) -> Path:
    """Precision-recall curves for the three models on the held-out 2025 concepts.

    Each panel shows the PR curve of one model (trained on 2022–2024, evaluated
    on 2025) alongside its no-skill baseline (the test-set success prevalence,
    shown as a dashed line) and its average precision (AP).
    """
    output_dir = Path(output_dir) if output_dir is not None else log_writer.FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ("Survey", modelling.SURVEY_MEASURES, "#2c7fb8"),
        ("Survey + behavioural", modelling.BEHAVIOURAL_CV_PREDICTORS, "#31a354"),
        (
            "Survey + behavioural + implicit",
            modelling.IMPLICIT_CV_PREDICTORS,
            "#e6550d",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), squeeze=False)

    for idx, (label, predictors, colour) in enumerate(models):
        res = modelling.logistic_train_test(df, predictors)
        y = res["y_test"]
        prob = res["y_prob_test"]

        precision, recall, _ = precision_recall_curve(y, prob)
        ap = float(average_precision_score(y, prob))
        prevalence = float(y.mean())

        ax = axes[0, idx]
        ax.step(recall, precision, where="post", color=colour, linewidth=2)
        ax.fill_between(recall, precision, step="post", color=colour, alpha=0.15)
        ax.axhline(prevalence, color="grey", linestyle="--", linewidth=1)
        ax.text(
            0.55,
            0.05,
            f"AP = {ap:.3f}\nno-skill = {prevalence:.3f}\nn = {res['n_test']}",
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{label} (n = {res['n_test']})", fontsize=11)
        ax.set_xlabel("Recall", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Precision", fontsize=10)

    fig.suptitle(
        "Precision-recall on the held-out 2025 concepts (train 2022–2024)",
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_calibration(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    filename: str = "calibration.png",
) -> Path:
    """Calibration curves for the three models on the held-out 2025 concepts.

    Two rows of panels — uniform-width binning (top) and quantile binning
    (bottom) — with one column per model. Each panel plots the model's mean
    predicted probability against the observed fraction of positives, alongside
    the diagonal marking perfect calibration, and annotates the Brier score.
    """
    output_dir = Path(output_dir) if output_dir is not None else log_writer.FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ("Survey", modelling.SURVEY_MEASURES, "#2c7fb8"),
        ("Survey + behavioural", modelling.BEHAVIOURAL_CV_PREDICTORS, "#31a354"),
        (
            "Survey + behavioural + implicit",
            modelling.IMPLICIT_CV_PREDICTORS,
            "#e6550d",
        ),
    ]
    strategies = [
        ("Uniform binning", "uniform", "o", "-"),
        ("Quantile binning", "quantile", "s", "--"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), squeeze=False)

    for row, (strat_title, strat_name, marker, linestyle) in enumerate(strategies):
        for col, (label, predictors, colour) in enumerate(models):
            res = modelling.logistic_train_test(df, predictors)
            y = res["y_test"]
            prob = res["y_prob_test"]

            fraction_pos, mean_pred = calibration_curve(
                y, prob, n_bins=5, strategy=strat_name
            )

            ax = axes[row, col]
            ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
            ax.plot(
                mean_pred,
                fraction_pos,
                marker=marker,
                color=colour,
                linestyle=linestyle,
                linewidth=2,
            )
            ax.text(
                0.03,
                0.92,
                f"Brier score = {res['test_brier']:.3f}\nn = {res['n_test']}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal", adjustable="box")
            if row == 0:
                ax.set_title(f"{label} (n = {res['n_test']})", fontsize=11)
            if row == 1:
                ax.set_xlabel("Mean predicted probability", fontsize=10)
            if col == 0:
                ax.set_ylabel(
                    f"{strat_title}\nObserved success frequency", fontsize=10
                )

    fig.suptitle(
        "Calibration on the held-out 2025 concepts (train 2022–2024)",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_propensity_overlap(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    filename: str = "propensity_overlap.png",
) -> Path:
    """Propensity-score overlap for Survey vs rich package, before and after weighting.

    Two panels: the raw (unweighted) propensity-score distributions, and the
    inverse-propensity-weighted distributions. Weighting shifts the two groups'
    score distributions toward overlap, visually confirming that the measured
    confounders are balanced after adjustment.
    """
    output_dir = Path(output_dir) if output_dir is not None else log_writer.FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    res = modelling.propensity_ipw(df)
    e = res["e"].loc[res["outcome_index"]]
    treat = res["treat"].loc[res["outcome_index"]]
    w = res["weights"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), squeeze=False)
    bins = np.linspace(0, 1, 21)

    for group, colour, label in ((0, "#2c7fb8", "Survey"), (1, "#e6550d", "Rich package")):
        mask = treat == group
        g_e = e[mask]
        g_w = w[mask]
        axes[0, 0].hist(g_e, bins=bins, alpha=0.5, color=colour, label=label, density=True)
        axes[0, 1].hist(
            g_e, bins=bins, weights=g_w, alpha=0.5, color=colour, label=label, density=True
        )

    axes[0, 0].set_title("Before weighting", fontsize=11)
    axes[0, 1].set_title("After weighting (inverse probability)", fontsize=11)
    for ax in axes[0]:
        ax.set_xlabel("Propensity score (P(rich package))", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle(
        "Propensity-score overlap: Survey vs rich package",
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cost_benefit_ev(
    df: pd.DataFrame | None = None,
    *,
    result: dict | None = None,
    output_dir=None,
    filename: str = "cost_benefit_ev.png",
) -> Path:
    """Expected cost per concept against the decision threshold, both screens.

    Uses the recalibrated repeated-K-fold predictions from
    :func:`modelling.cost_benefit_analysis`. The analytical cost-minimising
    threshold ``t*`` (base case) is marked with a vertical line; lower cost is
    better (equivalently higher expected value).
    """
    if result is None:
        result = modelling.cost_benefit_analysis(df)
    output_dir = Path(output_dir) if output_dir is not None else log_writer.FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = result["thresholds"]
    base = result["base"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        thresholds, base["cost_survey"], color="#2c7fb8", lw=2, label="Survey screen"
    )
    ax.plot(
        thresholds,
        base["cost_behavioural"],
        color="#e6550d",
        lw=2,
        label="Behavioural screen",
    )
    ax.axvline(base["t_star"], color="0.35", ls="--", lw=1, alpha=0.8)
    ax.text(
        base["t_star"] + 0.02,
        0.96,
        f"t* = {base['t_star']:.2f}",
        transform=ax.get_xaxis_transform(),
        va="top",
        ha="left",
        fontsize=10,
    )
    # Mark the empirical (sweep) minima of each screen.
    minima = [
        (base["empirical_argmin_survey"], base["empirical_cost_survey"], "#2c7fb8", "survey min"),
        (
            base["empirical_argmin_behavioural"],
            base["empirical_cost_behavioural"],
            "#e6550d",
            "behavioural min",
        ),
    ]
    for tx, cost, colour, label in minima:
        ax.plot(
            [tx], [cost],
            marker="o", markersize=8, color=colour,
            mec="white", mew=1.4, zorder=6,
        )
        ax.annotate(
            f"{label} (t = {tx:.2f})",
            xy=(tx, cost),
            xytext=(14, 10),
            textcoords="offset points",
            fontsize=9,
            color="black",
            arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
        )
    ax.set_xlabel("Decision threshold t (launch if p >= t)")
    ax.set_ylabel("Expected cost per concept (EUR)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlim(0, 1)
    ax.set_title("Cost-benefit: expected cost vs threshold (recalibrated predictions)")
    ax.legend(frameon=False)
    fig.tight_layout()

    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cost_benefit_sensitivity(
    df: pd.DataFrame | None = None,
    *,
    result: dict | None = None,
    sensitivity: "pd.DataFrame | None" = None,
    title: str = "Net value per concept (EUR) vs FP/FN costs",
    output_dir=None,
    filename: str = "cost_benefit_sensitivity.png",
) -> Path:
    """Heatmap of the net value per concept over the FP x FN cost grid.

    Each cell is the net value (decision saving minus the premium) at that
    (C_FP, C_FN) pair, evaluated at the pair's own ``t*``. Green is positive
    (worth it), red negative. Pass ``sensitivity`` directly to plot a different
    comparison's grid (e.g. the implicit screen).
    """
    if sensitivity is None:
        if result is None:
            result = modelling.cost_benefit_analysis(df)
        sensitivity = result["sensitivity"]
    output_dir = Path(output_dir) if output_dir is not None else log_writer.FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = sensitivity
    mat = grid.pivot(index="C_FP", columns="C_FN", values="net_per_concept").sort_index()

    fig, ax = plt.subplots(figsize=(9, 4.2))
    vmax = float(np.abs(mat.to_numpy()).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="RdYlGn", norm=norm)

    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([f"{c/1000:,.0f}k" for c in mat.columns], fontsize=9)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([f"{c/1000:,.0f}k" for c in mat.index], fontsize=9)
    ax.set_xlabel("C_FN - cost of stopping a winner (EUR)")
    ax.set_ylabel("C_FP - cost of launching a loser (EUR)")
    ax.set_title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.to_numpy()[i, j]
            ax.text(j, i, f"{v:+,.0f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Net value per concept (EUR)")
    fig.tight_layout()

    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    data = data_utilities.load_historical_data()
    full_path = plot_cross_phase_scatter(data)
    notable_path = plot_notable_cross_phase_scatter(data)
    fits_path = plot_regression_fits(data)
    ridge_path = plot_ridge_regression_fits(data)
    comparison_path = plot_regression_comparison(data)
    validation_path = plot_time_validation(data)
    cv_diffs_path = plot_repeated_cv_diffs(
        data,
        modelling.BEHAVIOURAL_CV_PREDICTORS,
        modelling.IMPLICIT_CV_PREDICTORS,
        "behavioural",
        "implicit",
        "Repeated K-fold CV: distribution of (implicit − behavioural) fold differences",
    )
    cv_diffs_survey_path = plot_repeated_cv_diffs(
        data,
        modelling.SURVEY_MEASURES,
        modelling.BEHAVIOURAL_CV_PREDICTORS,
        "survey",
        "behavioural",
        "Repeated K-fold CV: distribution of (behavioural − survey) fold differences",
        filename="repeated_cv_diffs_survey.png",
    )
    classification_cv_path = plot_classification_cv_diffs(data)
    precision_recall_path = plot_precision_recall(data)
    calibration_path = plot_calibration(data)
    propensity_path = plot_propensity_overlap(data)
    cost_benefit_ev_path = plot_cost_benefit_ev(data)
    cost_benefit_sensitivity_path = plot_cost_benefit_sensitivity(data)
    print(f"Full cross-phase grid saved to: {full_path}")
    print(f"Notable cross-phase grid saved to: {notable_path}")
    print(f"Regression fits saved to: {fits_path}")
    print(f"Ridge regression fits saved to: {ridge_path}")
    print(f"Regression comparison saved to: {comparison_path}")
    print(f"Time validation saved to: {validation_path}")
    print(f"Repeated CV diffs (implicit vs behavioural) saved to: {cv_diffs_path}")
    print(f"Repeated CV diffs (behavioural vs survey) saved to: {cv_diffs_survey_path}")
    print(f"Classification CV diffs saved to: {classification_cv_path}")
    print(f"Precision-recall curves saved to: {precision_recall_path}")
    print(f"Calibration curves saved to: {calibration_path}")
    print(f"Propensity-score overlap saved to: {propensity_path}")
    print(f"Cost-benefit EV saved to: {cost_benefit_ev_path}")
    print(f"Cost-benefit sensitivity saved to: {cost_benefit_sensitivity_path}")

