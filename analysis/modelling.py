"""Predictive modelling of launch outcomes from concept-test research.

Defines the modelling target — a binary success outcome derived from
``Sales_vs_Target_Pct`` — compares success between research packages, fits a
first linear model, and records the caveats that bound interpretation. Writes to
``output/logs/modelling.md``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

import data_utilities
import log_writer

#: Success threshold: ``Sales_vs_Target_Pct`` >= this value is a success.
SUCCESS_THRESHOLD: float = 100.0

#: Survey (stated) measures used as predictors in the first model.
SURVEY_MEASURES: list[str] = ["Stated_Appeal", "Purchase_Intent"]

#: Behavioural measure added as a predictor in the second model.
BEHAVIOURAL_MEASURES: list[str] = ["Behavioural_Choice_Pct"]

#: Implicit measure added as a predictor in the third model.
IMPLICIT_MEASURES: list[str] = ["Implicit_Score"]

#: Regression target for the first model.
TARGET: str = "Sales_vs_Target_Pct"

#: Regularisation strengths tried by cross-validation for the ridge models.
RIDGE_ALPHAS: list[float] = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

#: Predictor sets for the behavioural and implicit models compared in the
#: repeated K-fold CV. Both are restricted to the shared implicit subsample so
#: the only difference between them is the predictor set.
BEHAVIOURAL_CV_PREDICTORS: list[str] = SURVEY_MEASURES + BEHAVIOURAL_MEASURES
IMPLICIT_CV_PREDICTORS: list[str] = BEHAVIOURAL_CV_PREDICTORS + IMPLICIT_MEASURES

#: Fixed regularisation strength used by the "clean" comparison (option A),
#: where the two models differ only in their predictor set.
FIXED_ALPHA: float = 10.0

#: Regularisation strengths used to check option A's sensitivity to alpha.
ALPHA_SENSITIVITY: list[float] = [0.0, 10.0, 50.0]


def _context_facts(df: pd.DataFrame) -> dict[str, float | int]:
    """Compute the small set of data facts referenced by the caveats."""
    # Implicit_Score subsample (Combined packages only).
    implicit = df[df["Research_Package"] == "Combined"]
    implicit_complete = data_utilities.drop_missing_rows(implicit, ["Implicit_Score"])

    # Collinearity between the research-economics metrics.
    cost_days = df[["Research_Cost_EUR", "Turnaround_Days"]].apply(
        pd.to_numeric, errors="coerce"
    )

    # Package-mix shift 2022 -> 2025.
    package_pct = (
        pd.crosstab(df["Test_Year"], df["Research_Package"])
        .div(pd.crosstab(df["Test_Year"], df["Research_Package"]).sum(axis=1), axis=0)
        * 100
    )

    return {
        "n_implicit_applicable": len(implicit),
        "n_implicit_complete": len(implicit_complete),
        "implicit_missing_rate": (len(implicit) - len(implicit_complete))
        / len(implicit)
        * 100,
        "r_cost_days": float(cost_days.corr().iloc[0, 1]),
        "survey_2022": package_pct.loc[2022, "Survey"],
        "survey_2025": package_pct.loc[2025, "Survey"],
    }


def _package_success(df: pd.DataFrame) -> dict:
    """Compare binary success across research packages (Survey baseline).

    Restricted to launched concepts with an observed ``Sales_vs_Target_Pct``.
    """
    launched = df[df["Launched"] == 1].copy()
    launched["sales"] = pd.to_numeric(launched["Sales_vs_Target_Pct"], errors="coerce")
    sub = launched[launched["sales"].notna()].copy()
    sub["success"] = (sub["sales"] >= SUCCESS_THRESHOLD).astype(int)

    ct = pd.crosstab(sub["Research_Package"], sub["success"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)

    success_rate = ct[1] / ct.sum(axis=1) * 100
    n = ct.sum(axis=1)

    cost = launched.groupby("Research_Package", observed=True)["Research_Cost_EUR"].apply(
        lambda s: pd.to_numeric(s, errors="coerce").mean()
    )
    days = launched.groupby("Research_Package", observed=True)["Turnaround_Days"].apply(
        lambda s: pd.to_numeric(s, errors="coerce").mean()
    )

    base_success = success_rate["Survey"]
    base_cost = cost["Survey"]
    base_days = days["Survey"]

    rows = []
    for package in ["Survey", "Behavioural", "Combined"]:
        rows.append(
            {
                "Research package": package,
                "n": int(n[package]),
                "Success rate (%)": f"{success_rate[package]:.1f}",
                "Δ success vs Survey (pp)": (
                    "—" if package == "Survey" else f"{success_rate[package] - base_success:+.1f}"
                ),
                "Δ cost vs Survey (€)": (
                    "—" if package == "Survey" else f"{cost[package] - base_cost:+,.0f}"
                ),
                "Δ turnaround vs Survey (days)": (
                    "—" if package == "Survey" else f"{days[package] - base_days:+.1f}"
                ),
            }
        )

    return {
        "table": pd.DataFrame(rows),
        "n_launched": int(len(launched)),
        "n_complete": int(len(sub)),
        "n_missing_target": int(launched["sales"].isna().sum()),
        "overall_success": float(sub["success"].mean() * 100),
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "success_rate": {pkg: float(success_rate[pkg]) for pkg in ["Survey", "Behavioural", "Combined"]},
        "delta_pp": {pkg: float(success_rate[pkg] - base_success) for pkg in ["Behavioural", "Combined"]},
    }


def _applicable_mask(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Rows where every conditional column is applicable (not the NA marker)."""
    mask = pd.Series(True, index=df.index)
    for column in columns:
        if column in data_utilities.CONDITIONAL_COLUMNS:
            mask &= ~df[column].eq(data_utilities.NA_MARKER).fillna(False)
    return mask


def linear_model(
    df: pd.DataFrame,
    predictors: list[str],
    complete_on: list[str] | None = None,
) -> dict:
    """Fit a linear regression of the target on ``predictors``.

    Restricted to launched concepts (the target is launch-phase only), to rows
    where the ``complete_on`` columns are applicable, and to per-model complete
    cases on ``complete_on`` and the target. ``complete_on`` defaults to
    ``predictors``; pass it explicitly to refit a model on a fixed subset.
    """
    if complete_on is None:
        complete_on = predictors
    launched = df[df["Launched"] == 1]
    launched = launched[_applicable_mask(launched, complete_on)]
    sub = data_utilities.drop_missing_rows(launched, complete_on + [TARGET])

    X = pd.DataFrame(
        {col: pd.to_numeric(sub[col], errors="coerce") for col in predictors}
    )
    y = pd.to_numeric(sub[TARGET], errors="coerce")

    model = LinearRegression().fit(X, y)
    n = len(X)
    k = len(predictors)
    r2 = float(model.score(X, y))
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    coefs = pd.DataFrame(
        [{"Term": "Intercept", "Coefficient": float(model.intercept_)}]
        + [
            {"Term": col, "Coefficient": float(coef)}
            for col, coef in zip(predictors, model.coef_)
        ]
    )

    return {
        "n": n,
        "r2": r2,
        "adj_r2": adj_r2,
        "coefs": coefs,
        "coef": dict(zip(predictors, model.coef_.tolist())),
        "y": y.to_numpy(),
        "y_pred": model.predict(X),
    }


def ridge_model(
    df: pd.DataFrame,
    predictors: list[str],
    complete_on: list[str] | None = None,
    alpha: float | None = None,
) -> dict:
    """Fit a ridge regression of the target on ``predictors``.

    Predictors are standardised to z-scores so the L2 penalty treats each
    variable equally. When ``alpha`` is ``None`` (default) the regularisation
    strength is selected by leave-one-out cross-validation (``RidgeCV``);
    otherwise a fixed ``alpha`` is used. Passing ``alpha=0.0`` recovers an
    unregularised OLS fit on the standardised predictors, giving per-SD
    coefficients directly comparable with the ridge ones. The sample is
    restricted exactly like :func:`linear_model`.
    """
    if complete_on is None:
        complete_on = predictors
    launched = df[df["Launched"] == 1]
    launched = launched[_applicable_mask(launched, complete_on)]
    sub = data_utilities.drop_missing_rows(launched, complete_on + [TARGET])

    X = pd.DataFrame(
        {col: pd.to_numeric(sub[col], errors="coerce") for col in predictors}
    )
    y = pd.to_numeric(sub[TARGET], errors="coerce")

    scaler = StandardScaler().fit(X)
    X_std = pd.DataFrame(scaler.transform(X), columns=predictors)

    if alpha is None:
        model = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_std, y)
    else:
        model = Ridge(alpha=alpha).fit(X_std, y)
    n = len(X)
    r2 = float(model.score(X_std, y))

    coefs = pd.DataFrame(
        [{"Term": "Intercept", "Coefficient": float(model.intercept_)}]
        + [
            {"Term": col, "Coefficient": float(coef)}
            for col, coef in zip(predictors, model.coef_)
        ]
    )

    return {
        "n": n,
        "r2": r2,
        "alpha": float(model.alpha_) if alpha is None else float(alpha),
        "coefs": coefs,
        "coef": dict(zip(predictors, model.coef_.tolist())),
        "y": y.to_numpy(),
        "y_pred": model.predict(X_std),
    }


def _prepare_xy(
    df: pd.DataFrame, predictors: list[str], years: list[int]
) -> tuple[pd.DataFrame, pd.Series]:
    """Restrict to launched concepts in ``years`` with applicable + complete data.

    Returns the numeric predictor matrix ``X`` and target ``y``. Structural
    ``"NA"`` markers are excluded via the applicability mask, then genuinely
    missing values (``pd.NA``) are dropped per-model on the predictors and the
    target.
    """
    sub = df[(df["Test_Year"].isin(years)) & (df["Launched"] == 1)]
    sub = sub[_applicable_mask(sub, predictors)]
    sub = data_utilities.drop_missing_rows(sub, predictors + [TARGET])
    X = pd.DataFrame(
        {col: pd.to_numeric(sub[col], errors="coerce") for col in predictors}
    )
    y = pd.to_numeric(sub[TARGET], errors="coerce")
    return X, y


def ridge_train_test(
    df: pd.DataFrame,
    predictors: list[str],
    train_years: tuple[int, ...] = (2022, 2023, 2024),
    test_year: int = 2025,
    alpha: float | None = None,
) -> dict:
    """Fit a ridge regression on ``train_years`` and predict ``test_year``.

    The scaler and the regularisation strength (cross-validated via ``RidgeCV``
    when ``alpha`` is ``None``) are determined from the training data only, then
    applied unchanged to the held-out year, so the test-year metrics are
    genuinely out of sample. Returns training/test sizes, the selected alpha,
    in-sample (train) R², out-of-sample (test) R², RMSE and MAE, the fitted
    coefficients, and the test-year observed and predicted values.
    """
    X_train, y_train = _prepare_xy(df, predictors, list(train_years))
    X_test, y_test = _prepare_xy(df, predictors, [test_year])

    scaler = StandardScaler().fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    if alpha is None:
        model = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_train_std, y_train)
    else:
        model = Ridge(alpha=alpha).fit(X_train_std, y_train)

    y_pred = model.predict(X_test_std)

    ss_res = float(((y_test - y_pred) ** 2).sum())
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
    r2_test = 1 - ss_res / ss_tot
    rmse = float(np.sqrt(((y_test.to_numpy() - y_pred) ** 2).mean()))
    mae = float(np.abs(y_test.to_numpy() - y_pred).mean())

    coefs = pd.DataFrame(
        [{"Term": "Intercept", "Coefficient": float(model.intercept_)}]
        + [
            {"Term": col, "Coefficient": float(coef)}
            for col, coef in zip(predictors, model.coef_)
        ]
    )

    return {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "alpha": float(model.alpha_) if alpha is None else float(alpha),
        "r2_train": float(model.score(X_train_std, y_train)),
        "r2_test": float(r2_test),
        "rmse": rmse,
        "mae": mae,
        "coefs": coefs,
        "y_test": y_test.to_numpy(),
        "y_pred": y_pred,
    }


def _shared_subset(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Cases complete on ``predictors`` and the target, applicable for all.

    Used so both models in a pairwise CV comparison are fit on identical cases.
    """
    launched = df[df["Launched"] == 1]
    launched = launched[_applicable_mask(launched, predictors)]
    return data_utilities.drop_missing_rows(launched, predictors + [TARGET])


def _fold_metrics(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    predictors: list[str],
    alpha: float | None,
) -> dict:
    """Fit a ridge model on one CV fold and score it on the held-out fold.

    ``alpha`` is used directly, or selected by leave-one-out ``RidgeCV`` when
    ``None``. The scaler is fit on the training fold only. Returns out-of-sample
    R², RMSE and MAE.
    """
    Xtr = X_train[predictors]
    Xte = X_test[predictors]
    scaler = StandardScaler().fit(Xtr)
    Xtr_std = scaler.transform(Xtr)
    Xte_std = scaler.transform(Xte)

    if alpha is None:
        model = RidgeCV(alphas=RIDGE_ALPHAS).fit(Xtr_std, y_train)
    else:
        model = Ridge(alpha=alpha).fit(Xtr_std, y_train)

    y_pred = model.predict(Xte_std)
    ss_res = float(((y_test - y_pred) ** 2).sum())
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    rmse = float(np.sqrt(((y_test - y_pred) ** 2).mean()))
    mae = float(np.abs(y_test - y_pred).mean())
    return {"r2": r2, "rmse": rmse, "mae": mae}


def repeated_kfold_compare(
    df: pd.DataFrame,
    base_predictors: list[str],
    full_predictors: list[str],
    alpha: float | None = None,
    n_splits: int = 7,
    n_repeats: int = 20,
    random_state: int = 42,
) -> dict:
    """Repeated K-fold comparison of two nested ridge models.

    The base model uses ``base_predictors`` and the full model uses
    ``full_predictors`` (a superset). Both are fit on identical folds of the
    shared subset (cases complete on ``full_predictors``), so within-fold
    differences isolate the contribution of the added predictors. With ``alpha``
    fixed (option A) the two models differ only in their predictor set; with
    ``alpha=None`` (option B) each model selects its own strength by
    leave-one-out ``RidgeCV``. Returns fold-level differences (full − base) in
    R², RMSE and MAE, plus each model's per-fold R².
    """
    sub = _shared_subset(df, full_predictors)
    X = pd.DataFrame(
        {
            col: pd.to_numeric(sub[col], errors="coerce")
            for col in full_predictors
        }
    )
    y = pd.to_numeric(sub[TARGET], errors="coerce").to_numpy()

    rkf = RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    r2_diff, rmse_diff, mae_diff = [], [], []
    r2_base, r2_full = [], []
    for train_idx, test_idx in rkf.split(X):
        base = _fold_metrics(
            X.iloc[train_idx],
            y[train_idx],
            X.iloc[test_idx],
            y[test_idx],
            base_predictors,
            alpha,
        )
        full = _fold_metrics(
            X.iloc[train_idx],
            y[train_idx],
            X.iloc[test_idx],
            y[test_idx],
            full_predictors,
            alpha,
        )
        r2_diff.append(full["r2"] - base["r2"])
        rmse_diff.append(full["rmse"] - base["rmse"])
        mae_diff.append(full["mae"] - base["mae"])
        r2_base.append(base["r2"])
        r2_full.append(full["r2"])

    return {
        "n_cases": int(len(sub)),
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "n_folds": n_splits * n_repeats,
        "alpha": None if alpha is None else float(alpha),
        "r2_diff": np.asarray(r2_diff),
        "rmse_diff": np.asarray(rmse_diff),
        "mae_diff": np.asarray(mae_diff),
        "r2_base": np.asarray(r2_base),
        "r2_full": np.asarray(r2_full),
    }


def summarise_fold_diffs(diffs: np.ndarray) -> dict:
    """Summarise an array of fold differences (full − base).

    Returns the mean, standard deviation, median, 2.5th/97.5th-percentile
    confidence interval, and the proportion of folds where the difference is
    positive (full better for R²; subtract from one for RMSE/MAE).
    """
    diffs = np.asarray(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean": float(diffs.mean()),
        "sd": float(diffs.std()),
        "median": float(np.median(diffs)),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_positive": float((diffs > 0).mean()),
    }


def diff_summary_table(res: dict, better_label: str = "Full") -> pd.DataFrame:
    """Build the metric summary table for a repeated-K-fold result."""
    rows = []
    for metric, key, lower_is_better in [
        ("Δ R²", "r2_diff", False),
        ("Δ RMSE", "rmse_diff", True),
        ("Δ MAE", "mae_diff", True),
    ]:
        s = summarise_fold_diffs(res[key])
        better = s["p_positive"] if not lower_is_better else 1 - s["p_positive"]
        rows.append(
            {
                "Metric": metric,
                "Mean": f"{s['mean']:+.3f}",
                "SD": f"{s['sd']:.3f}",
                "Median": f"{s['median']:+.3f}",
                "95% CI": f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]",
                f"{better_label} better (% of folds)": f"{better * 100:.0f}",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Write the modelling log: target, package comparison, and caveats."""
    df = data_utilities.load_historical_data()
    facts = _context_facts(df)
    ps = _package_success(df)

    log = log_writer.MarkdownLog("modelling")
    with log:
        log.header("Modelling — Target, Package Comparison, and Caveats")

        # 1. Model target ---------------------------------------------------
        log.heading("1. Model target", level=2)
        log.paragraph(
            "The success outcome is binary, derived from "
            "``Sales_vs_Target_Pct``: **success** when the value is at least "
            f"{SUCCESS_THRESHOLD:.0f}, and **failure** otherwise. It is defined "
            "only for launched concepts (``Launched == 1``); of the "
            f"{ps['n_launched']} launched concepts, {ps['n_missing_target']} lack "
            "``Sales_vs_Target_Pct`` and are excluded (per-model deletion on the "
            f"target), leaving {ps['n_complete']} observations at an overall "
            f"success rate of {ps['overall_success']:.1f}%."
        )

        # 2. Success by research package -------------------------------------
        log.heading("2. Success by research package", level=2)
        log.paragraph(
            "Success is compared across packages to assess whether the extra "
            "cost and turnaround of the behavioural and combined (implicit) "
            "packages are reflected in higher success rates. Survey is the "
            "baseline; the difference columns are relative to Survey."
        )
        log.table(ps["table"])
        log.paragraph(
            "Success rises with research intensity: Survey "
            f"{ps['success_rate']['Survey']:.1f}% → Behavioural "
            f"{ps['success_rate']['Behavioural']:.1f}% → Combined "
            f"{ps['success_rate']['Combined']:.1f}% "
            f"(χ² = {ps['chi2']:.2f}, df = {ps['dof']}, p = {ps['p']:.3f}). "
            "The Behavioural package buys a "
            f"{ps['delta_pp']['Behavioural']:+.1f} pp success uplift over "
            "Survey, and the Combined package buys "
            f"{ps['delta_pp']['Combined']:+.1f} pp. This is consistent with the "
            "behavioural/implicit measures adding predictive value, but the "
            "comparison is observational (see caveats) and not evidence of "
            "causation."
        )

        # 3. Caveats and limitations ------------------------------------------
        log.heading("3. Caveats and limitations", level=2)
        log.paragraph(
            "**Predictive, not causal.** Packages were selected by teams "
            "according to need and resources, not randomised, so the package "
            "comparison supports prediction, not causation. A randomised pilot "
            "is required for causal claims."
        )
        log.paragraph(
            "**Outcome selection.** Outcomes exist only for launched concepts, "
            "and launch is pre-filtered by ``Recommendation`` (sometimes "
            "overridden). Results generalise to the launched sub-population "
            "only."
        )
        log.paragraph(
            "**Small ``Implicit_Score`` subsample.** Only "
            f"{facts['n_implicit_complete']} of {facts['n_implicit_applicable']} "
            f"Combined concepts ({facts['implicit_missing_rate']:.1f}% missing) "
            "have ``Implicit_Score``, and the missingness mechanism is "
            "unconfirmed; treat coefficients on it cautiously."
        )
        log.paragraph(
            "**Non-stationary package mix.** Survey fell from "
            f"{facts['survey_2022']:.0f}% to {facts['survey_2025']:.0f}% over "
            "2022–2025. Use a time-ordered split (train 2022–2024, test 2025) "
            "to avoid leakage and reflect current conditions."
        )
        log.paragraph(
            "**Feature availability and ``\"NA\"`` handling.** Behavioural/"
            "implicit metrics exist only for their packages; ``\"NA\"`` markers "
            "are absent-by-design and must be excluded from features and "
            "targets, or the model leaks launch/package information."
        )
        log.paragraph(
            "**Collinearity and modest signal.** Cost and turnaround correlate "
            f"r ≈ {facts['r_cost_days']:.2f} and track ``Launch_Support_EUR``; "
            "the strongest cross-phase signal is moderate (r ≈ 0.4–0.5), so "
            "expect modest performance — report uncertainty, not point "
            "estimates."
        )
        log.paragraph(
            "**Balance, costs, and scope.** Check class balance per target "
            "(avoid accuracy alone); costs are nominal euros; single-context "
            "data may not generalise."
        )

        # 4. First model — linear regression on survey measures -------------
        lm = linear_model(df, SURVEY_MEASURES)
        log.heading("4. First model — linear regression on survey measures", level=2)
        log.paragraph(
            "A first, straightforward model regresses ``Sales_vs_Target_Pct`` "
            "on the survey (stated) measures ``Stated_Appeal`` and "
            "``Purchase_Intent`` using scikit-learn's ``LinearRegression``. It "
            "is fit on launched concepts only (the target is launch-phase), "
            "after per-model deletion on the two predictors and the target."
        )
        log.table_pair(
            lm["coefs"],
            [
                ["Observations (n)", lm["n"]],
                ["R²", f"{lm['r2']:.3f}"],
                ["Adjusted R²", f"{lm['adj_r2']:.3f}"],
            ],
            right_headers=["Key", "Value"],
            float_format="{:.3f}".format,
        )
        log.paragraph(
            f"The model explains only {lm['r2'] * 100:.1f}% of the variance in "
            "``Sales_vs_Target_Pct``. A one-point increase in "
            f"``Stated_Appeal`` is associated with a "
            f"{lm['coef']['Stated_Appeal']:+.2f} pp change and a one-point "
            f"increase in ``Purchase_Intent`` with "
            f"{lm['coef']['Purchase_Intent']:+.2f} pp. The fit is weak, "
            "consistent with the earlier finding that stated measures carry "
            "little cross-phase predictive signal. This is an in-sample fit; "
            "out-of-sample evaluation on a time-ordered holdout is the next "
            "step."
        )

        # 5. Second model — adding the behavioural measure ------------------
        bh_model = linear_model(df, SURVEY_MEASURES + BEHAVIOURAL_MEASURES)
        bh_baseline = linear_model(
            df, SURVEY_MEASURES, complete_on=SURVEY_MEASURES + BEHAVIOURAL_MEASURES
        )
        log.heading("5. Second model — adding the behavioural measure", level=2)
        log.paragraph(
            "``Behavioural_Choice_Pct`` is added to the survey measures. "
            "Because it is collected only for Behavioural and Combined "
            "packages, this model is restricted to that subset; the "
            "survey-only baseline is refit on the same complete cases so the "
            "increment is comparable."
        )
        log.table_pair(
            bh_model["coefs"],
            [
                ["Observations (n)", bh_model["n"]],
                ["Baseline R² (same subset)", f"{bh_baseline['r2']:.3f}"],
                ["Model R²", f"{bh_model['r2']:.3f}"],
                ["Model adjusted R²", f"{bh_model['adj_r2']:.3f}"],
            ],
            right_headers=["Key", "Value"],
            float_format="{:.3f}".format,
        )
        log.paragraph(
            "Adding ``Behavioural_Choice_Pct`` raises R² from "
            f"{bh_baseline['r2'] * 100:.1f}% to {bh_model['r2'] * 100:.1f}% "
            f"({(bh_model['r2'] - bh_baseline['r2']) * 100:+.1f} pp). Each +1 "
            "pp in ``Behavioural_Choice_Pct`` is associated with a "
            f"{bh_model['coef']['Behavioural_Choice_Pct']:+.2f} pp change in "
            "``Sales_vs_Target_Pct``, consistent with the exploratory finding "
            "that the behavioural measure is a stronger predictor than the "
            "stated measures. The sample is smaller (Behavioural/Combined "
            "concepts only) and this remains an in-sample fit."
        )

        # 6. Third model — adding the implicit measure ----------------------
        imp_model = linear_model(
            df, SURVEY_MEASURES + BEHAVIOURAL_MEASURES + IMPLICIT_MEASURES
        )
        imp_base = linear_model(
            df,
            SURVEY_MEASURES + BEHAVIOURAL_MEASURES,
            complete_on=SURVEY_MEASURES + BEHAVIOURAL_MEASURES + IMPLICIT_MEASURES,
        )
        log.heading("6. Third model — adding the implicit measure", level=2)
        log.paragraph(
            "``Implicit_Score`` is added to the survey and behavioural "
            "measures. Because it is collected only for Combined packages, "
            "this model is restricted to that subset; the survey-plus-"
            "behavioural baseline is refit on the same complete cases so the "
            "increment is comparable."
        )
        log.table_pair(
            imp_model["coefs"],
            [
                ["Observations (n)", imp_model["n"]],
                ["Baseline R² (same subset)", f"{imp_base['r2']:.3f}"],
                ["Model R²", f"{imp_model['r2']:.3f}"],
                ["Model adjusted R²", f"{imp_model['adj_r2']:.3f}"],
            ],
            right_headers=["Key", "Value"],
            float_format="{:.3f}".format,
        )
        log.paragraph(
            "Adding ``Implicit_Score`` raises R² from "
            f"{imp_base['r2'] * 100:.1f}% to {imp_model['r2'] * 100:.1f}% "
            f"({(imp_model['r2'] - imp_base['r2']) * 100:+.1f} pp), a smaller "
            "increment than the behavioural measure. The sample is very small "
            f"(n = {imp_model['n']}, Combined launched concepts only) and "
            "``Purchase_Intent``'s coefficient flips sign, a symptom of "
            "collinearity among the four measures; individual coefficients "
            "should be read cautiously, consistent with the ``Implicit_Score`` "
            "small-subsample and collinearity caveats."
        )

        # 7. Visualising the model fits --------------------------------------
        # Deferred import avoids a circular dependency: visualisation imports
        # this module for the model-fitting helpers.
        import visualisation

        fig_path = visualisation.plot_regression_fits(df)
        fig_rel = os.path.relpath(fig_path, log.path.parent).replace(os.sep, "/")

        log.heading("7. Visualising the model fits", level=2)
        log.paragraph(
            "The three regressions are shown side by side as observed-vs-"
            "fitted scatter plots, produced by "
            "``visualisation.plot_regression_fits``. Each panel plots the "
            "model's predicted ``Sales_vs_Target_Pct`` against the observed "
            "value, with the dashed identity line marking a perfect "
            "prediction, and is annotated with its R² and sample size. The "
            "panels share identical axis limits so the spread of points "
            "relative to the diagonal is directly comparable across models. "
            "The usable sample shrinks across the panels because each model is "
            "restricted to the concepts for which its predictors exist, as "
            "noted in sections 4–6."
        )
        log.image("Observed vs fitted Sales_vs_Target_Pct by model", fig_rel)

        # 8. Ridge regression — controlling for collinearity -----------------
        log.heading("8. Ridge regression — controlling for collinearity", level=2)
        log.paragraph(
            "The exploration phase showed the research measures are "
            "inter-correlated, and this collinearity caused "
            "``Purchase_Intent``'s coefficient to flip sign in the implicit "
            "model. The same three-stage regression is therefore refit with "
            "ridge regression (scikit-learn's ``RidgeCV``) to control for "
            "collinearity. Predictors are standardised to z-scores so the L2 "
            "penalty treats each variable equally, and the regularisation "
            "strength (alpha) is selected by leave-one-out cross-validation. "
            "Coefficients are reported per +1 standard deviation of each "
            "predictor, so they are not directly comparable to the raw-unit "
            "coefficients in sections 4–6."
        )

        ridge_specs = [
            ("Survey", SURVEY_MEASURES),
            ("Survey + behavioural", SURVEY_MEASURES + BEHAVIOURAL_MEASURES),
            (
                "Survey + behavioural + implicit",
                SURVEY_MEASURES + BEHAVIOURAL_MEASURES + IMPLICIT_MEASURES,
            ),
        ]
        ridge_fits = {
            title: ridge_model(df, predictors) for title, predictors in ridge_specs
        }
        for title, fit in ridge_fits.items():
            log.heading(title, level=3)
            log.table_pair(
                fit["coefs"],
                [
                    ["Observations (n)", fit["n"]],
                    ["Alpha", f"{fit['alpha']:.1f}"],
                    ["R²", f"{fit['r2']:.3f}"],
                ],
                right_headers=["Key", "Value"],
                float_format="{:.3f}".format,
            )

        imp_ridge = ridge_fits["Survey + behavioural + implicit"]
        log.paragraph(
            "Ridge regularisation stabilises the coefficients: "
            "``Purchase_Intent`` is now positive in all three models (unlike "
            "its negative OLS coefficient in the implicit model), confirming "
            "the sign flip was a collinearity artefact. The in-sample R² is "
            "slightly lower than OLS — for the implicit model "
            f"{imp_ridge['r2']:.3f} versus {imp_model['r2']:.3f} — the "
            "expected cost of shrinkage in exchange for more stable, "
            "better-conditioned estimates."
        )

        ridge_fig_path = visualisation.plot_ridge_regression_fits(df)
        ridge_fig_rel = os.path.relpath(ridge_fig_path, log.path.parent).replace(
            os.sep, "/"
        )
        log.paragraph(
            "The ridge fits are shown below in the same observed-vs-fitted "
            "format as section 7, for a direct visual comparison with the OLS "
            "panels. The predictions are in the original "
            "``Sales_vs_Target_Pct`` units — only the predictors were "
            "standardised, not the target."
        )
        log.image(
            "Observed vs fitted Sales_vs_Target_Pct by model (ridge)",
            ridge_fig_rel,
        )

        # 9. OLS vs ridge comparison ------------------------------------------
        log.heading("9. OLS vs ridge comparison", level=2)
        log.paragraph(
            "The OLS and ridge models are compared directly. Ridge "
            "coefficients are on the standardised (per-SD) scale, so the OLS "
            "coefficients are recomputed on the same standardised predictors "
            "(``ridge_model(..., alpha=0.0)``) to make the two directly "
            "comparable; on this scale the intercept is the mean target at the "
            "predictor means. R² is scale-invariant and therefore matches the "
            "raw-unit OLS values in sections 4–6."
        )

        comparison_specs = [
            ("Survey", SURVEY_MEASURES),
            ("Survey + behavioural", SURVEY_MEASURES + BEHAVIOURAL_MEASURES),
            (
                "Survey + behavioural + implicit",
                SURVEY_MEASURES + BEHAVIOURAL_MEASURES + IMPLICIT_MEASURES,
            ),
        ]
        ols_fits = {
            title: ridge_model(df, predictors, alpha=0.0)
            for title, predictors in comparison_specs
        }
        ridge_fits = {
            title: ridge_model(df, predictors)
            for title, predictors in comparison_specs
        }

        log.heading("Fit metrics", level=3)
        log.table(
            pd.DataFrame(
                [
                    {
                        "Model": title,
                        "n": ols_fits[title]["n"],
                        "OLS R²": f"{ols_fits[title]['r2']:.3f}",
                        "Ridge R²": f"{ridge_fits[title]['r2']:.3f}",
                        "Alpha": f"{ridge_fits[title]['alpha']:.1f}",
                    }
                    for title, _ in comparison_specs
                ]
            )
        )

        for title, _ in comparison_specs:
            ols_coef = ols_fits[title]["coefs"].set_index("Term")["Coefficient"]
            ridge_coef = ridge_fits[title]["coefs"].set_index("Term")["Coefficient"]
            terms = list(ols_coef.index)
            comparison = pd.DataFrame(
                {
                    "Term": terms,
                    "OLS (per SD)": [ols_coef[t] for t in terms],
                    "Ridge (per SD)": [ridge_coef[t] for t in terms],
                }
            )
            log.heading(title, level=3)
            log.table(comparison, float_format="{:.3f}".format)

        log.paragraph(
            "Ridge shrinks every coefficient toward zero relative to OLS and, "
            "most importantly, stabilises ``Purchase_Intent`` — negative in the "
            "OLS implicit model but positive in every ridge model — confirming "
            "the sign flip was a collinearity artefact. The fit-metrics table "
            "shows the in-sample R² drops only slightly under ridge, the modest "
            "cost of the stabilisation."
        )

        comp_fig_path = visualisation.plot_regression_comparison(df)
        comp_fig_rel = os.path.relpath(comp_fig_path, log.path.parent).replace(
            os.sep, "/"
        )
        log.paragraph(
            "The OLS (top row) and ridge (bottom row) fits are shown below on "
            "identical axis limits, so the spread of points relative to the "
            "identity line is directly comparable across all six panels."
        )
        log.image(
            "OLS vs ridge: observed vs fitted Sales_vs_Target_Pct",
            comp_fig_rel,
        )

        # 10. Time-ordered validation ---------------------------------------
        log.heading("10. Time-ordered validation: train 2022–2024, test 2025", level=2)
        log.paragraph(
            "To obtain an honest out-of-sample estimate, the ridge regressions are "
            "refit on 2022–2024 data only and used to predict the held-out 2025 "
            "concepts. The scaler and the cross-validated regularisation strength "
            "(alpha) are derived from the training years alone and applied "
            "unchanged to 2025, so nothing from the test year leaks into the "
            "fit. This split also respects the non-stationary package mix noted "
            "in the caveats: 2025 is the year in which the behavioural and "
            "implicit packages are most common, so it is the strictest test of "
            "whether the model generalises."
        )
        validation_specs = comparison_specs
        validation_rows = []
        for title, predictors in validation_specs:
            result = ridge_train_test(df, predictors)
            validation_rows.append(
                {
                    "Model": title,
                    "n train": result["n_train"],
                    "n test": result["n_test"],
                    "Alpha": f"{result['alpha']:.1f}",
                    "Train R²": f"{result['r2_train']:.3f}",
                    "Test R²": f"{result['r2_test']:.3f}",
                    "RMSE": f"{result['rmse']:.1f}",
                    "MAE": f"{result['mae']:.1f}",
                }
            )
            log.heading(title, level=3)
            log.table(result["coefs"], float_format="{:.3f}".format)

        log.heading("Out-of-sample metrics", level=3)
        log.table(pd.DataFrame(validation_rows))

        log.paragraph(
            "The test-year R² is the honest measure of predictive value. The "
            "survey-only model does not generalise — its test R² is negative, "
            "meaning it predicts 2025 no better (in fact slightly worse) than "
            "simply using the test-year mean. Adding the behavioural measure is "
            "the step that genuinely helps: it is the only model with a positive "
            "out-of-sample R². Adding the implicit measure on top is counter-"
            "productive in this split — despite the highest in-sample fit (train "
            "R² = 0.288), its test R² is negative because it is fit on only "
            "47 training concepts (23 in the test year), the smallest subsample "
            "of the three, so the extra predictor overfits and fails to "
            "generalise to 2025. This is exactly the small-``Implicit_Score``-"
            "subsample risk flagged in the caveats, and it cautions against "
            "reading the implicit model's in-sample gains as real predictive "
            "value."
        )

        validation_fig_path = visualisation.plot_time_validation(df)
        validation_fig_rel = os.path.relpath(
            validation_fig_path, log.path.parent
        ).replace(os.sep, "/")
        log.paragraph(
            "The out-of-sample predictions are shown below in the same "
            "observed-vs-predicted format as sections 7–9. Each panel is "
            "annotated with the test-year R² and the number of 2025 test "
            "observations; points are the held-out 2025 concepts, not the "
            "training data used to fit the models."
        )
        log.image(
            "Time-ordered validation: observed vs predicted Sales_vs_Target_Pct (2025)",
            validation_fig_rel,
        )

        # 11. Repeated K-fold CV — isolating the implicit measure -------------
        log.heading(
            "11. Repeated K-fold CV — isolating the implicit measure's contribution",
            level=2,
        )
        log.paragraph(
            "A single 2025 hold-out (23 concepts) cannot tell whether the implicit "
            "model's failure reflects a hard fold or too few points. Repeated "
            "K-fold cross-validation replaces that one split with many random "
            "splits of the shared 70-case implicit subsample. On every fold the "
            "behavioural and implicit models are fit on the *same* training cases "
            "and evaluated on the *same* test cases, and the within-fold "
            "difference (implicit − behavioural) is recorded. Because the folds "
            "are paired, any overall difficulty of a fold cancels out, and the "
            "distribution of differences isolates what ``Implicit_Score`` adds. "
            "Two regularisation strategies are compared: a fixed α (option A), so "
            "the only difference between the models is the predictor set, and a "
            "per-fold cross-validated α (option B), which pits the best-tuned "
            "implicit model against the best-tuned behavioural model."
        )

        option_a = repeated_kfold_compare(
            df, BEHAVIOURAL_CV_PREDICTORS, IMPLICIT_CV_PREDICTORS, alpha=FIXED_ALPHA
        )
        option_b = repeated_kfold_compare(
            df, BEHAVIOURAL_CV_PREDICTORS, IMPLICIT_CV_PREDICTORS, alpha=None
        )
        a_r2 = summarise_fold_diffs(option_a["r2_diff"])
        b_r2 = summarise_fold_diffs(option_b["r2_diff"])

        log.heading("Option A — fixed regularisation (α = 10)", level=3)
        log.paragraph(
            f"Both models share the same fixed α = {FIXED_ALPHA:.0f}, so the only "
            "difference between them is the predictor set. Over "
            f"{option_a['n_folds']} folds the behavioural model's fold R² averages "
            f"{option_a['r2_base'].mean():.3f} and the implicit model's "
            f"{option_a['r2_full'].mean():.3f}. These fold-level R² values are "
            "negative on average only because each test fold holds ~10 points, so "
            "the per-fold R² is very noisy; the paired Δ, which cancels this "
            "fold-to-fold noise, is the quantity that matters."
        )
        log.table(diff_summary_table(option_a, "Implicit"))

        log.heading("Option A — sensitivity to α", level=3)
        sensitivity_rows = []
        for alpha in ALPHA_SENSITIVITY:
            res = repeated_kfold_compare(
                df, BEHAVIOURAL_CV_PREDICTORS, IMPLICIT_CV_PREDICTORS, alpha=alpha
            )
            s = summarise_fold_diffs(res["r2_diff"])
            sensitivity_rows.append(
                {
                    "α": f"{alpha:.0f}",
                    "Mean ΔR²": f"{s['mean']:+.3f}",
                    "95% CI": f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]",
                    "Implicit better (% of folds)": f"{s['p_positive'] * 100:.0f}",
                }
            )
        log.table(pd.DataFrame(sensitivity_rows))

        log.heading("Option B — per-fold cross-validated α", level=3)
        log.paragraph(
            "Each model selects its own α by leave-one-out cross-validation on the "
            "training fold, so this compares the best-tuned implicit model against "
            "the best-tuned behavioural model."
        )
        log.table(diff_summary_table(option_b, "Implicit"))

        log.paragraph(
            f"Both strategies tell the same story. Under option A the mean ΔR² is "
            f"{a_r2['mean']:+.3f} with a 95% percentile interval of "
            f"[{a_r2['ci_lo']:+.3f}, {a_r2['ci_hi']:+.3f}], and the implicit model "
            f"beats the behavioural model on {a_r2['p_positive'] * 100:.0f}% of "
            f"folds; under option B the mean ΔR² is {b_r2['mean']:+.3f} with "
            f"interval [{b_r2['ci_lo']:+.3f}, {b_r2['ci_hi']:+.3f}]. Because option "
            "A and option B agree in direction and magnitude, the conclusion is "
            "not an artefact of fine-tuning the regularisation. The RMSE and MAE "
            "differences point the same way: the implicit model's average "
            "out-of-sample error is lower (negative Δ), so it is not simply "
            "trading variance in R² for a worse absolute fit."
        )
        log.paragraph(
            "Read against the single time-ordered split, this is the sample-size "
            "explanation, not a difficult-fold artefact. Where the 2025 hold-out "
            "put the implicit model's test R² at −0.150 against the behavioural "
            "model's +0.183 — a Δ of about −0.33 — the 140 random folds give a "
            "mean ΔR² of roughly +0.05 to +0.07, positive on ~70% of folds. The "
            "single negative result is therefore the product of one small, hard "
            "fold rather than evidence that the implicit measure *hurts* "
            "prediction. At the same time, every 95% interval still straddles "
            "zero, so the 70-case sample is too small to confirm that the "
            "positive effect is real. That is precisely the case for a pilot "
            "study: the point estimate favours the implicit measure, but "
            "confirming it requires more implicit data than the current sample "
            "provides."
        )

        cv_fig_path = visualisation.plot_repeated_cv_diffs(
            df,
            BEHAVIOURAL_CV_PREDICTORS,
            IMPLICIT_CV_PREDICTORS,
            "behavioural",
            "implicit",
            "Repeated K-fold CV: distribution of (implicit − behavioural) fold differences",
        )
        cv_fig_rel = os.path.relpath(cv_fig_path, log.path.parent).replace(os.sep, "/")
        log.image(
            "Repeated K-fold CV: distribution of (implicit − behavioural) fold differences",
            cv_fig_rel,
        )

        # 12. Repeated K-fold CV — behavioural vs survey ----------------------
        log.heading(
            "12. Repeated K-fold CV — behavioural vs survey",
            level=2,
        )
        log.paragraph(
            "The same paired repeated K-fold design is applied one step earlier, to "
            "ask whether ``Behavioural_Choice_Pct`` adds predictive value over the "
            "survey-only model. Both models are restricted to the shared behavioural "
            "subsample — the launched concepts for which ``Behavioural_Choice_Pct`` "
            "is applicable (Behavioural and Combined packages) — so, as before, the "
            "only difference between the two models is the predictor set, and any "
            "fold difficulty cancels out in the paired difference."
        )

        beh_a = repeated_kfold_compare(
            df, SURVEY_MEASURES, BEHAVIOURAL_CV_PREDICTORS, alpha=FIXED_ALPHA
        )
        beh_b = repeated_kfold_compare(
            df, SURVEY_MEASURES, BEHAVIOURAL_CV_PREDICTORS, alpha=None
        )
        beh_a_r2 = summarise_fold_diffs(beh_a["r2_diff"])
        beh_b_r2 = summarise_fold_diffs(beh_b["r2_diff"])

        log.heading("Option A — fixed regularisation (α = 10)", level=3)
        log.paragraph(
            f"Over {beh_a['n_folds']} folds on the {beh_a['n_cases']}-case behavioural "
            f"subsample, the survey model's fold R² averages "
            f"{beh_a['r2_base'].mean():.3f} and the behavioural model's "
            f"{beh_a['r2_full'].mean():.3f}."
        )
        log.table(diff_summary_table(beh_a, "Behavioural"))

        log.heading("Option A — sensitivity to α", level=3)
        sensitivity_rows = []
        for alpha in ALPHA_SENSITIVITY:
            res = repeated_kfold_compare(
                df, SURVEY_MEASURES, BEHAVIOURAL_CV_PREDICTORS, alpha=alpha
            )
            s = summarise_fold_diffs(res["r2_diff"])
            sensitivity_rows.append(
                {
                    "α": f"{alpha:.0f}",
                    "Mean ΔR²": f"{s['mean']:+.3f}",
                    "95% CI": f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]",
                    "Behavioural better (% of folds)": f"{s['p_positive'] * 100:.0f}",
                }
            )
        log.table(pd.DataFrame(sensitivity_rows))

        log.heading("Option B — per-fold cross-validated α", level=3)
        log.table(diff_summary_table(beh_b, "Behavioural"))

        log.paragraph(
            f"Both strategies again agree. The mean ΔR² is {beh_a_r2['mean']:+.3f} "
            f"(option A, 95% interval [{beh_a_r2['ci_lo']:+.3f}, "
            f"{beh_a_r2['ci_hi']:+.3f}]) and {beh_b_r2['mean']:+.3f} (option B, "
            f"interval [{beh_b_r2['ci_lo']:+.3f}, {beh_b_r2['ci_hi']:+.3f}]), with "
            f"the behavioural model beating the survey model on "
            f"{beh_a_r2['p_positive'] * 100:.0f}% of folds, and the RMSE/MAE "
            "differences pointing the same way. The behavioural advantage is "
            "therefore robust to how regularisation is set."
        )
        log.paragraph(
            "The behavioural measure shows a clear, consistent positive effect — "
            "larger in magnitude and on a much larger sample (204 vs 70 cases) than "
            "the implicit measure's — yet its 95% interval still straddles zero. "
            "That is the important cross-check: even the behavioural measure, whose "
            "value the time-ordered split supported (test R² +0.183 vs −0.071), "
            "cannot be confirmed as statistically significant under repeated "
            "resampling at the current sample size. The implicit measure's failure "
            "to reach significance in section 11 is therefore not evidence against "
            "it — it fails the same test the behavioural measure fails — but rather "
            "a sample-size limitation that a pilot study is designed to resolve."
        )

        beh_fig_path = visualisation.plot_repeated_cv_diffs(
            df,
            SURVEY_MEASURES,
            BEHAVIOURAL_CV_PREDICTORS,
            "survey",
            "behavioural",
            "Repeated K-fold CV: distribution of (behavioural − survey) fold differences",
            filename="repeated_cv_diffs_survey.png",
        )
        beh_fig_rel = os.path.relpath(beh_fig_path, log.path.parent).replace(os.sep, "/")
        log.image(
            "Repeated K-fold CV: distribution of (behavioural − survey) fold differences",
            beh_fig_rel,
        )

    print(f"Modelling log written to: {log.path}")


if __name__ == "__main__":
    main()

