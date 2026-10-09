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
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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

#: Inverse regularisation strength for the logistic classification models
#: (``LogisticRegression`` with an L2 penalty).
LOGISTIC_C: float = 1.0

#: Confounders used to model research-package selection (the propensity score).
#: ``Sample_Size`` is excluded because the exploratory analysis found it
#: negligibly correlated with launch outcomes, and ``Research_Cost_EUR`` /
#: ``Turnaround_Days`` are excluded because they are consequences of the package
#: choice rather than confounders.
PROPENSITY_CONFOUNDERS: list[str] = ["Test_Year", "Category", "Innovation_Type"]

#: Package levels treated as the "rich" research package, versus Survey.
RICH_PACKAGES: list[str] = ["Behavioural", "Combined"]

#: Inverse regularisation strength for the propensity-score model.
PROPENSITY_C: float = 1.0

#: Quantiles at which the stabilised inverse-propensity weights are trimmed.
WEIGHT_TRIM_QUANTILES: tuple[float, float] = (0.01, 0.99)

#: Decision-threshold grid for the cost-benefit confusion-matrix sweep.
COST_BENEFIT_THRESHOLD_STEP: float = 0.05

#: Quantiles of ``Launch_Support_EUR`` used for the false-positive cost in the
#: cost-benefit analysis: Q1 (low), median (central), Q3 (high).
COST_BENEFIT_FP_QUANTILES: tuple[float, float, float] = (0.25, 0.5, 0.75)

#: False-negative cost levels for the sensitivity analysis, expressed as
#: multiples of the median ``Launch_Support_EUR``. Q1/median/Q3 are added
#: alongside these multipliers.
COST_BENEFIT_FN_MULTIPLIERS: tuple[float, ...] = (1.25, 1.5, 2.0, 3.0)

#: Plausible false-negative cost, as a multiple of the median launch spend. A
#: launched winner's profit margin typically exceeds the launch spend, so the
#: missed profit of a stopped winner is set to 1.5x the false-positive cost
#: rather than the break-even 1x.
PLAUSIBLE_FN_TO_FP_RATIO: float = 1.5

#: Number of cluster-bootstrap resamples for the net-value confidence interval.
COST_BENEFIT_BOOTSTRAP_N: int = 1000


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


def logistic_model(
    df: pd.DataFrame,
    predictors: list[str],
    complete_on: list[str] | None = None,
    C: float = LOGISTIC_C,
) -> dict:
    """Fit a regularised logistic regression predicting binary success.

    Success is ``Sales_vs_Target_Pct >= SUCCESS_THRESHOLD``. The model uses an
    L2 penalty with balanced class weights (``LogisticRegression``), predictors
    standardised to z-scores, and ``C`` as the inverse regularisation strength.
    The sample is restricted exactly like :func:`linear_model`. Coefficients
    are reported per standard deviation (log-odds and odds ratios); metrics are
    in-sample and therefore optimistic.
    """
    if complete_on is None:
        complete_on = predictors
    launched = df[df["Launched"] == 1]
    launched = launched[_applicable_mask(launched, complete_on)]
    sub = data_utilities.drop_missing_rows(launched, complete_on + [TARGET])

    X = pd.DataFrame(
        {col: pd.to_numeric(sub[col], errors="coerce") for col in predictors}
    )
    y_raw = pd.to_numeric(sub[TARGET], errors="coerce")
    y = (y_raw >= SUCCESS_THRESHOLD).astype(int)

    scaler = StandardScaler().fit(X)
    X_std = pd.DataFrame(scaler.transform(X), columns=predictors)

    model = LogisticRegression(
        C=C,
        l1_ratio=0.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
    ).fit(X_std, y)

    y_prob = model.predict_proba(X_std)[:, 1]
    y_pred = model.predict(X_std)

    coefs = pd.DataFrame(
        [
            {
                "Term": "Intercept",
                "Coefficient": float(model.intercept_[0]),
                "Odds ratio": float(np.exp(model.intercept_[0])),
            }
        ]
        + [
            {
                "Term": col,
                "Coefficient": float(coef),
                "Odds ratio": float(np.exp(coef)),
            }
            for col, coef in zip(predictors, model.coef_[0])
        ]
    )

    return {
        "n": len(X),
        "success_rate": float(y.mean() * 100),
        "coefs": coefs,
        "accuracy": float(accuracy_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
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


def logistic_train_test(
    df: pd.DataFrame,
    predictors: list[str],
    train_years: tuple[int, ...] = (2022, 2023, 2024),
    test_year: int = 2025,
    C: float = LOGISTIC_C,
) -> dict:
    """Fit a regularised logistic regression on ``train_years``, predict ``test_year``.

    Success is ``Sales_vs_Target_Pct >= SUCCESS_THRESHOLD``. The scaler is fit
    on the training data only and applied unchanged to the held-out year, so the
    test-year metrics are genuinely out of sample. Returns training/test sizes
    and success rates, the fitted coefficients (odds ratios), in-sample (train)
    and out-of-sample (test) classification metrics.
    """
    X_train, y_train_cont = _prepare_xy(df, predictors, list(train_years))
    X_test, y_test_cont = _prepare_xy(df, predictors, [test_year])

    y_train = (y_train_cont >= SUCCESS_THRESHOLD).astype(int)
    y_test = (y_test_cont >= SUCCESS_THRESHOLD).astype(int)

    scaler = StandardScaler().fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    model = LogisticRegression(
        C=C,
        l1_ratio=0.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
    ).fit(X_train_std, y_train)

    y_prob_train = model.predict_proba(X_train_std)[:, 1]
    y_pred_train = model.predict(X_train_std)
    y_prob_test = model.predict_proba(X_test_std)[:, 1]
    y_pred_test = model.predict(X_test_std)

    coefs = pd.DataFrame(
        [
            {
                "Term": "Intercept",
                "Coefficient": float(model.intercept_[0]),
                "Odds ratio": float(np.exp(model.intercept_[0])),
            }
        ]
        + [
            {
                "Term": col,
                "Coefficient": float(coef),
                "Odds ratio": float(np.exp(coef)),
            }
            for col, coef in zip(predictors, model.coef_[0])
        ]
    )

    return {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_success_rate": float(y_train.mean() * 100),
        "test_success_rate": float(y_test.mean() * 100),
        "coefs": coefs,
        "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
        "train_balanced_accuracy": float(balanced_accuracy_score(y_train, y_pred_train)),
        "train_roc_auc": float(roc_auc_score(y_train, y_prob_train)),
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred_test)),
        "test_precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred_test, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob_test)),
        "test_brier": float(brier_score_loss(y_test, y_prob_test)),
        "y_test": y_test.to_numpy(),
        "y_prob_test": y_prob_test,
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


def _fold_classify(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    predictors: list[str],
    C: float,
) -> dict:
    """Fit a logistic model on one CV fold and score it on the held-out fold.

    Returns out-of-sample ROC-AUC, balanced accuracy and F1. ROC-AUC is NaN if
    the test fold holds a single class.
    """
    Xtr = X_train[predictors]
    Xte = X_test[predictors]
    scaler = StandardScaler().fit(Xtr)
    Xtr_std = scaler.transform(Xtr)
    Xte_std = scaler.transform(Xte)

    model = LogisticRegression(
        C=C,
        l1_ratio=0.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
    ).fit(Xtr_std, y_train)

    y_prob = model.predict_proba(Xte_std)[:, 1]
    y_pred = model.predict(Xte_std)

    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        auc = float("nan")

    return {
        "auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }


def repeated_kfold_compare_logistic(
    df: pd.DataFrame,
    base_predictors: list[str],
    full_predictors: list[str],
    C: float = LOGISTIC_C,
    n_splits: int = 7,
    n_repeats: int = 20,
    random_state: int = 42,
) -> dict:
    """Paired repeated K-fold comparison of two nested logistic models.

    Mirrors :func:`repeated_kfold_compare` for classification. Both models are
    fit on identical folds of the shared subset (cases complete on
    ``full_predictors``), so within-fold differences (full − base) isolate the
    contribution of the added predictors. Returns fold-level differences in
    ROC-AUC, balanced accuracy and F1 (all higher is better), plus each model's
    per-fold ROC-AUC.
    """
    sub = _shared_subset(df, full_predictors)
    X = pd.DataFrame(
        {
            col: pd.to_numeric(sub[col], errors="coerce")
            for col in full_predictors
        }
    )
    y = (
        pd.to_numeric(sub[TARGET], errors="coerce") >= SUCCESS_THRESHOLD
    ).astype(int).to_numpy()

    rkf = RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    auc_diff, bal_diff, f1_diff = [], [], []
    auc_base, auc_full = [], []
    for train_idx, test_idx in rkf.split(X):
        base = _fold_classify(
            X.iloc[train_idx],
            y[train_idx],
            X.iloc[test_idx],
            y[test_idx],
            base_predictors,
            C,
        )
        full = _fold_classify(
            X.iloc[train_idx],
            y[train_idx],
            X.iloc[test_idx],
            y[test_idx],
            full_predictors,
            C,
        )
        if np.isnan(base["auc"]) or np.isnan(full["auc"]):
            auc_diff.append(np.nan)
        else:
            auc_diff.append(full["auc"] - base["auc"])
        bal_diff.append(full["balanced_accuracy"] - base["balanced_accuracy"])
        f1_diff.append(full["f1"] - base["f1"])
        auc_base.append(base["auc"])
        auc_full.append(full["auc"])

    auc_diff = np.asarray(auc_diff, dtype=float)
    valid = ~np.isnan(auc_diff)

    return {
        "n_cases": int(len(sub)),
        "n_folds": n_splits * n_repeats,
        "n_folds_auc": int(valid.sum()),
        "auc_diff": auc_diff[valid],
        "bal_diff": np.asarray(bal_diff),
        "f1_diff": np.asarray(f1_diff),
        "auc_base": np.asarray(auc_base),
        "auc_full": np.asarray(auc_full),
    }


def classification_diff_summary_table(
    res: dict, better_label: str = "Full"
) -> pd.DataFrame:
    """Build the metric summary table for a repeated-K-fold classification result.

    All three metrics are higher-is-better, so the "better" share is the
    proportion of folds with a positive difference.
    """
    rows = []
    for metric, key in [
        ("Δ ROC-AUC", "auc_diff"),
        ("Δ balanced accuracy", "bal_diff"),
        ("Δ F1", "f1_diff"),
    ]:
        s = summarise_fold_diffs(res[key])
        rows.append(
            {
                "Metric": metric,
                "Mean": f"{s['mean']:+.3f}",
                "SD": f"{s['sd']:.3f}",
                "Median": f"{s['median']:+.3f}",
                "95% CI": f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]",
                f"{better_label} better (% of folds)": f"{s['p_positive'] * 100:.0f}",
            }
        )
    return pd.DataFrame(rows)


def _propensity_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot encode the propensity confounders and build the rich-package target."""
    treat = df["Research_Package"].isin(RICH_PACKAGES).astype(int)
    parts = [df[col].astype(str).rename(col) for col in PROPENSITY_CONFOUNDERS]
    X = pd.get_dummies(pd.concat(parts, axis=1), drop_first=True).astype(float)
    return X, treat


def propensity_scores(df: pd.DataFrame) -> dict:
    """Fit the package-selection propensity model on all concepts.

    Returns the one-hot design matrix, the binary treatment, the fitted
    propensity scores, and the model coefficients (log-odds and odds ratios).
    """
    X, treat = _propensity_features(df)
    model = LogisticRegression(
        C=PROPENSITY_C,
        l1_ratio=0.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
    ).fit(X, treat)
    e = pd.Series(model.predict_proba(X)[:, 1], index=df.index)

    coefs = pd.DataFrame(
        [
            {
                "Term": "Intercept",
                "Coefficient": float(model.intercept_[0]),
                "Odds ratio": float(np.exp(model.intercept_[0])),
            }
        ]
        + [
            {
                "Term": col,
                "Coefficient": float(coef),
                "Odds ratio": float(np.exp(coef)),
            }
            for col, coef in zip(X.columns, model.coef_[0])
        ]
    )
    return {"X": X, "treat": treat, "e": e, "model": model, "coefs": coefs}


def _smd(x1: np.ndarray, x0: np.ndarray) -> float:
    """Standardised mean difference (pooled SD) between two samples."""
    m1, m0 = float(np.mean(x1)), float(np.mean(x0))
    v1, v0 = float(np.var(x1, ddof=1)), float(np.var(x0, ddof=1))
    denom = np.sqrt((v1 + v0) / 2.0)
    return (m1 - m0) / denom if denom > 0 else 0.0


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.average(x, weights=w))


def _wvar(x: np.ndarray, w: np.ndarray) -> float:
    m = _wmean(x, w)
    return float(np.average((x - m) ** 2, weights=w))


def _wsmd(x1: np.ndarray, x0: np.ndarray, w1: np.ndarray, w0: np.ndarray) -> float:
    """Weighted standardised mean difference between two samples."""
    m1, m0 = _wmean(x1, w1), _wmean(x0, w0)
    v1, v0 = _wvar(x1, w1), _wvar(x0, w0)
    denom = np.sqrt((v1 + v0) / 2.0)
    return (m1 - m0) / denom if denom > 0 else 0.0


def propensity_ipw(df: pd.DataFrame) -> dict:
    """Estimate the rich-package success effect via inverse propensity weighting.

    The propensity model is fit on the full population; the outcome effect is
    estimated on the launched concepts with a recorded ``Sales_vs_Target_Pct``.
    Stabilised ATE weights (``P(T=1)/e`` and ``P(T=0)/(1-e)``) are trimmed at the
    1st/99th percentile, and the Hajek (self-normalising) estimator gives the
    weighted success rates. Balance is summarised with standardised mean
    differences before and after weighting.
    """
    ps = propensity_scores(df)
    X, treat, e = ps["X"], ps["treat"], ps["e"]

    outcome = (df["Launched"] == 1) & df["Sales_vs_Target_Pct"].notna()
    idx = df.index[outcome]

    treat_out = treat.loc[idx].to_numpy()
    y = (
        pd.to_numeric(df["Sales_vs_Target_Pct"].loc[idx], errors="coerce")
        >= SUCCESS_THRESHOLD
    ).astype(float).to_numpy()
    e_out = e.loc[idx].to_numpy()

    p1 = float(treat_out.mean())
    w = np.where(treat_out == 1, p1 / e_out, (1.0 - p1) / (1.0 - e_out))
    lo, hi = np.quantile(w, WEIGHT_TRIM_QUANTILES)
    n_trimmed = int(np.sum((w < lo) | (w > hi)))
    w = np.clip(w, lo, hi)

    def rates(t: np.ndarray, y_: np.ndarray, w_: np.ndarray) -> tuple[float, float, float]:
        n1 = float(w_[t == 1].sum())
        n0 = float(w_[t == 0].sum())
        r1 = float((y_[t == 1] * w_[t == 1]).sum() / n1) if n1 > 0 else float("nan")
        r0 = float((y_[t == 0] * w_[t == 0]).sum() / n0) if n0 > 0 else float("nan")
        return r0, r1, r1 - r0

    r0_raw, r1_raw, ate_raw = rates(treat_out, y, np.ones_like(w))
    r0_w, r1_w, ate_w = rates(treat_out, y, w)

    rows = []
    for col in X.columns:
        x = X.loc[idx, col].to_numpy()
        x1 = x[treat_out == 1]
        x0 = x[treat_out == 0]
        w1 = w[treat_out == 1]
        w0 = w[treat_out == 0]
        rows.append(
            {
                "Confounder": col,
                "SMD before": round(_smd(x1, x0), 3),
                "SMD after": round(_wsmd(x1, x0, w1, w0), 3),
            }
        )
    balance = pd.DataFrame(rows)

    ps_rows = []
    for g, label in ((0, "Survey"), (1, "Rich package")):
        g_e = e_out[treat_out == g]
        ps_rows.append(
            {
                "Group": label,
                "n": int(len(g_e)),
                "Mean PS": round(float(g_e.mean()), 3),
                "Min PS": round(float(g_e.min()), 3),
                "Max PS": round(float(g_e.max()), 3),
            }
        )
    ps_summary = pd.DataFrame(ps_rows)

    return {
        "coefs": ps["coefs"],
        "features": list(X.columns),
        "n_full": int(len(df)),
        "n_outcome": int(len(idx)),
        "p_treat": p1,
        "n_trimmed": n_trimmed,
        "ps_summary": ps_summary,
        "balance": balance,
        "unweighted": {"survey": r0_raw, "rich": r1_raw, "ate": ate_raw},
        "weighted": {"survey": r0_w, "rich": r1_w, "ate": ate_w},
        "e": e,
        "treat": treat,
        "outcome_index": idx,
        "weights": pd.Series(w, index=idx),
    }


def bootstrap_ipw_ate(df: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> dict:
    """Bootstrap the whole IPW pipeline to get a 95% CI for the ATE.

    Each bootstrap iteration resamples the full dataset, refits the propensity
    model and re-estimates the weighted ATE, so the CI reflects both sampling
    and propensity-estimation uncertainty (percentile method).
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    ates = []
    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot = df.iloc[sample_idx].reset_index(drop=True)
        ates.append(propensity_ipw(boot)["weighted"]["ate"])
    ates = np.array(ates)
    lo, hi = np.percentile(ates, [2.5, 97.5])
    return {
        "mean": float(ates.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": n_boot,
    }


def _fold_probabilities(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    predictors: list[str],
    C: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a logistic model on one fold; return train and test success probabilities."""
    Xtr = X_train[predictors]
    Xte = X_test[predictors]
    scaler = StandardScaler().fit(Xtr)
    model = LogisticRegression(
        C=C,
        l1_ratio=0.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
    ).fit(scaler.transform(Xtr), y_train)
    p_train = model.predict_proba(scaler.transform(Xtr))[:, 1]
    p_test = model.predict_proba(scaler.transform(Xte))[:, 1]
    return p_train, p_test


def _platt_calibrate(
    p_train: np.ndarray, y_train: np.ndarray, p_test: np.ndarray
) -> np.ndarray:
    """Platt-scale ``p_test`` via a sigmoid fit of ``y`` on ``logit(p_train)``.

    The balanced class weights of the base model shift the intercept so the raw
    probabilities are systematically overconfident; the sigmoid maps them back
    to the empirical base rate without touching the held-out fold.
    """
    eps = 1e-12
    lo = np.clip(p_train, eps, 1 - eps)
    lt = np.clip(p_test, eps, 1 - eps)
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(
        np.log(lo / (1 - lo)).reshape(-1, 1), y_train
    )
    return cal.predict_proba(np.log(lt / (1 - lt)).reshape(-1, 1))[:, 1]


def cost_benefit_oof(
    df: pd.DataFrame,
    base_predictors: list[str] = SURVEY_MEASURES,
    full_predictors: list[str] = BEHAVIOURAL_CV_PREDICTORS,
    C: float = LOGISTIC_C,
    n_splits: int = 7,
    n_repeats: int = 20,
    random_state: int = 42,
) -> dict:
    """Collect out-of-fold success probabilities for the survey and behavioural screens.

    Both models are fit on identical folds of the shared behavioural-capable
    subset (cases complete on ``full_predictors``), matching the repeated
    K-fold design. Raw probabilities are Platt-recalibrated within each fold
    before collection. Returns concatenated held-out labels and probabilities,
    Brier scores, and the base rate.
    """
    sub = _shared_subset(df, full_predictors)
    X = pd.DataFrame(
        {col: pd.to_numeric(sub[col], errors="coerce") for col in full_predictors}
    )
    y = (
        pd.to_numeric(sub[TARGET], errors="coerce") >= SUCCESS_THRESHOLD
    ).astype(int).to_numpy()

    rkf = RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    y_oof, p_base_raw, p_full_raw, p_base_cal, p_full_cal = [], [], [], [], []
    concept_idx: list[np.ndarray] = []
    for train_idx, test_idx in rkf.split(X):
        p_base_train, p_base_test = _fold_probabilities(
            X.iloc[train_idx], y[train_idx], X.iloc[test_idx], base_predictors, C
        )
        p_full_train, p_full_test = _fold_probabilities(
            X.iloc[train_idx], y[train_idx], X.iloc[test_idx], full_predictors, C
        )
        y_oof.append(y[test_idx])
        p_base_raw.append(p_base_test)
        p_full_raw.append(p_full_test)
        p_base_cal.append(_platt_calibrate(p_base_train, y[train_idx], p_base_test))
        p_full_cal.append(_platt_calibrate(p_full_train, y[train_idx], p_full_test))
        concept_idx.append(np.asarray(test_idx, dtype=int))

    y_oof = np.concatenate(y_oof)
    p_base_raw = np.concatenate(p_base_raw)
    p_full_raw = np.concatenate(p_full_raw)
    p_base_cal = np.concatenate(p_base_cal)
    p_full_cal = np.concatenate(p_full_cal)
    concept_idx = np.concatenate(concept_idx)

    return {
        "n_cases": int(len(sub)),
        "n_folds": n_splits * n_repeats,
        "n_oof": int(len(y_oof)),
        "base_rate": float(y_oof.mean()),
        "y": y_oof,
        "p_survey_raw": p_base_raw,
        "p_behavioural_raw": p_full_raw,
        "p_survey": p_base_cal,
        "p_behavioural": p_full_cal,
        "concept_idx": concept_idx,
        "brier_survey_raw": float(brier_score_loss(y_oof, p_base_raw)),
        "brier_behavioural_raw": float(brier_score_loss(y_oof, p_full_raw)),
        "brier_survey": float(brier_score_loss(y_oof, p_base_cal)),
        "brier_behavioural": float(brier_score_loss(y_oof, p_full_cal)),
    }


def confusion_sweep(
    y: np.ndarray, p: np.ndarray, thresholds: np.ndarray
) -> pd.DataFrame:
    """Tabulate TP/TN/FP/FN for every threshold in ``thresholds``.

    A case is predicted positive (launch) when its probability is ``>= t``.
    Counts are summed over all held-out folds.
    """
    rows = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        rows.append(
            {
                "t": float(t),
                "TP": int(np.sum((pred == 1) & (y == 1))),
                "FP": int(np.sum((pred == 1) & (y == 0))),
                "TN": int(np.sum((pred == 0) & (y == 0))),
                "FN": int(np.sum((pred == 0) & (y == 1))),
            }
        )
    return pd.DataFrame(rows)


def _expected_cost(
    sweep: pd.DataFrame, c_fp: float, c_fn: float, n: int
) -> np.ndarray:
    """Expected cost per concept at each threshold (TP and TN cost nothing)."""
    return (c_fp * sweep["FP"].to_numpy() + c_fn * sweep["FN"].to_numpy()) / n


def _cost_at(
    y: np.ndarray, p: np.ndarray, c_fp: float, c_fn: float, t: float, n: int
) -> float:
    """Expected cost per concept when thresholding the probabilities at exactly ``t``."""
    pred = (p >= t).astype(int)
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return (c_fp * fp + c_fn * fn) / n


def _optimal_threshold(c_fp: float, c_fn: float) -> float:
    """Cost-minimising threshold for a calibrated model, ``C_FP / (C_FP + C_FN)``."""
    return c_fp / (c_fp + c_fn)


def _auc_from_predictions(
    y: np.ndarray, p: np.ndarray, thresholds: np.ndarray, c_fp: float, c_fn: float
) -> float:
    """Area under the expected-cost curve for a single screen (threshold-free).

    Integrates the expected cost per concept over the threshold grid using the
    trapezoidal rule. Lower is better. Vectorised over thresholds so the
    cluster bootstrap can recompute it cheaply.
    """
    n = len(y)
    preds = (p[:, None] >= thresholds[None, :]).astype(int)
    fp = ((preds == 1) & (y[:, None] == 0)).sum(axis=0)
    fn = ((preds == 0) & (y[:, None] == 1)).sum(axis=0)
    costs = (c_fp * fp + c_fn * fn) / n
    return float(np.trapezoid(costs, thresholds))


def cost_benefit_analysis(
    df: pd.DataFrame, thresholds: np.ndarray | None = None
) -> dict:
    """Cost-benefit analysis of the behavioural screen versus the survey screen.

    Builds the cost model (FP = median ``Launch_Support_EUR``; FN = FP on the
    first pass because missed profit is unobserved), sweeps the decision
    threshold over the recalibrated repeated-K-fold predictions, converts each
    confusion matrix into an expected cost, and computes the net value of
    switching from the survey to the behavioural screen at the analytical
    optimum ``t*``. Also returns an interquartile range and a full FP x FN
    sensitivity grid.
    """
    if thresholds is None:
        thresholds = np.arange(
            0.0, 1.0 + COST_BENEFIT_THRESHOLD_STEP / 2, COST_BENEFIT_THRESHOLD_STEP
        )

    oof = cost_benefit_oof(df)
    sub = _shared_subset(df, BEHAVIOURAL_CV_PREDICTORS)
    ls = pd.to_numeric(sub["Launch_Support_EUR"], errors="coerce").dropna()
    q25, median, q75 = ls.quantile(list(COST_BENEFIT_FP_QUANTILES)).to_numpy()

    rc = pd.to_numeric(df["Research_Cost_EUR"], errors="coerce")
    survey_cost = float(rc[df["Research_Package"] == "Survey"].mean())
    behavioural_cost = float(rc[df["Research_Package"] == "Behavioural"].mean())
    premium = behavioural_cost - survey_cost

    sweep_survey = confusion_sweep(oof["y"], oof["p_survey"], thresholds)
    sweep_behavioural = confusion_sweep(oof["y"], oof["p_behavioural"], thresholds)

    n = oof["n_oof"]
    n_cases = oof["n_cases"]

    def net_value(c_fp: float, c_fn: float) -> dict:
        t_star = _optimal_threshold(c_fp, c_fn)
        cost_s = _expected_cost(sweep_survey, c_fp, c_fn, n)
        cost_b = _expected_cost(sweep_behavioural, c_fp, c_fn, n)
        cs = _cost_at(oof["y"], oof["p_survey"], c_fp, c_fn, t_star, n)
        cb_ = _cost_at(oof["y"], oof["p_behavioural"], c_fp, c_fn, t_star, n)
        decision_saving = cs - cb_
        net_per_concept = decision_saving - premium

        # Decompose the error saving at t* into its false-positive and
        # false-negative parts (counts are across all held-out folds).
        pred_s = (oof["p_survey"] >= t_star).astype(int)
        pred_b = (oof["p_behavioural"] >= t_star).astype(int)
        fp_s = int(np.sum((pred_s == 1) & (oof["y"] == 0)))
        fn_s = int(np.sum((pred_s == 0) & (oof["y"] == 1)))
        fp_b = int(np.sum((pred_b == 1) & (oof["y"] == 0)))
        fn_b = int(np.sum((pred_b == 0) & (oof["y"] == 1)))

        argmin_s = int(np.argmin(cost_s))
        argmin_b = int(np.argmin(cost_b))
        min_cost_s = float(cost_s[argmin_s])
        min_cost_b = float(cost_b[argmin_b])
        empirical_saving = min_cost_s - min_cost_b
        empirical_net_per_concept = empirical_saving - premium

        return {
            "c_fp": float(c_fp),
            "c_fn": float(c_fn),
            "t_star": float(t_star),
            "cost_survey": cost_s,
            "cost_behavioural": cost_b,
            "cost_survey_at_tstar": cs,
            "cost_behavioural_at_tstar": cb_,
            "empirical_argmin_survey": float(thresholds[argmin_s]),
            "empirical_argmin_behavioural": float(thresholds[argmin_b]),
            "empirical_cost_survey": min_cost_s,
            "empirical_cost_behavioural": min_cost_b,
            "empirical_decision_saving_per_concept": empirical_saving,
            "empirical_net_per_concept": empirical_net_per_concept,
            "decision_saving_per_concept": decision_saving,
            "net_per_concept": net_per_concept,
            "error_reduction": (fp_s + fn_s) - (fp_b + fn_b),
            "fp_survey": fp_s,
            "fn_survey": fn_s,
            "fp_behavioural": fp_b,
            "fn_behavioural": fn_b,
            "delta_fp": fp_b - fp_s,
            "delta_fn": fn_b - fn_s,
            "launches_survey": int(np.sum(pred_s == 1)),
            "launches_behavioural": int(np.sum(pred_b == 1)),
        }

    base = net_value(median, median)
    iqr_range = {"low": net_value(q25, q25), "high": net_value(q75, q75)}
    plausible = net_value(median, PLAUSIBLE_FN_TO_FP_RATIO * median)

    # Threshold-free summary: integrate the expected cost over the whole
    # threshold range. This averages out the single-threshold volatility of t*.
    auc_survey = float(np.trapezoid(base["cost_survey"], thresholds))
    auc_behavioural = float(np.trapezoid(base["cost_behavioural"], thresholds))
    auc = {
        "auc_survey": auc_survey,
        "auc_behavioural": auc_behavioural,
        "decision_saving_per_concept": auc_survey - auc_behavioural,
        "net_per_concept": (auc_survey - auc_behavioural) - premium,
    }

    fp_grid = [q25, median, q75]
    fn_grid = [q25, median, q75] + [m * median for m in COST_BENEFIT_FN_MULTIPLIERS]
    sensitivity_rows = []
    for c_fp in fp_grid:
        for c_fn in fn_grid:
            nv = net_value(c_fp, c_fn)
            sensitivity_rows.append(
                {
                    "C_FP": float(round(c_fp)),
                    "C_FN": float(round(c_fn)),
                    "t*": nv["t_star"],
                    "net_per_concept": nv["net_per_concept"],
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)

    # Break-even false-negative cost: the C_FN at which the net value crosses
    # zero (C_FP held at the median). A fine scan over the sensitivity range
    # locates the first (lower) crossing, interpolated linearly.
    be_fn_scan = np.linspace(q25, 3.0 * median, 500)
    be_net = np.array([net_value(median, c)["net_per_concept"] for c in be_fn_scan])
    break_even: dict | None = None
    for j in range(len(be_net) - 1):
        if be_net[j] < 0 <= be_net[j + 1]:
            frac = (0.0 - be_net[j]) / (be_net[j + 1] - be_net[j])
            be_fn = float(be_fn_scan[j] + frac * (be_fn_scan[j + 1] - be_fn_scan[j]))
            break_even = {
                "c_fp": float(median),
                "c_fn": be_fn,
                "ratio": be_fn / median,
            }
            break

    return {
        "n_cases": n_cases,
        "n_folds": oof["n_folds"],
        "n_oof": n,
        "base_rate": oof["base_rate"],
        "launch_support": {
            "q25": float(q25),
            "median": float(median),
            "q75": float(q75),
            "iqr": float(q75 - q25),
        },
        "premium": premium,
        "thresholds": thresholds,
        "calibration": {
            "brier_survey_raw": oof["brier_survey_raw"],
            "brier_behavioural_raw": oof["brier_behavioural_raw"],
            "brier_survey": oof["brier_survey"],
            "brier_behavioural": oof["brier_behavioural"],
        },
        "oof": {
            "y": oof["y"],
            "p_survey": oof["p_survey"],
            "p_behavioural": oof["p_behavioural"],
            "concept_idx": oof["concept_idx"],
        },
        "sweep": {"survey": sweep_survey, "behavioural": sweep_behavioural},
        "base": base,
        "auc": auc,
        "iqr_range": iqr_range,
        "plausible": plausible,
        "sensitivity": sensitivity,
        "break_even": break_even,
    }


def bootstrap_net_value(
    cb: dict, n_boot: int = COST_BENEFIT_BOOTSTRAP_N, seed: int = 0
) -> dict:
    """Cluster-bootstrap 95% CI for the base-case net value of switching.

    Resamples the shared-subset concepts (with replacement) and recomputes the
    net value at the base-case ``t*`` from the resampled out-of-fold
    predictions. Because each concept contributes ``n_repeats`` correlated
    predictions, resampling is done at the concept level. The behavioural
    premium is a recorded cost and is held fixed. Returns a percentile CI for
    the per-concept net value.
    """
    oof = cb["oof"]
    y = oof["y"]
    ps = oof["p_survey"]
    pb = oof["p_behavioural"]
    cidx = oof["concept_idx"]
    n_concepts = int(cidx.max()) + 1

    c_fp = cb["base"]["c_fp"]
    c_fn = cb["base"]["c_fn"]
    t_star = cb["base"]["t_star"]
    premium = cb["premium"]
    n_cases = cb["n_cases"]

    positions = [np.where(cidx == c)[0] for c in range(n_concepts)]
    rng = np.random.default_rng(seed)

    per_concept = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.integers(0, n_concepts, size=n_concepts)
        pos = np.concatenate([positions[c] for c in sample])
        yy = y[pos]
        pred_s = (ps[pos] >= t_star).astype(int)
        pred_b = (pb[pos] >= t_star).astype(int)
        fp_s = int(np.sum((pred_s == 1) & (yy == 0)))
        fn_s = int(np.sum((pred_s == 0) & (yy == 1)))
        fp_b = int(np.sum((pred_b == 1) & (yy == 0)))
        fn_b = int(np.sum((pred_b == 0) & (yy == 1)))

        n = len(yy)
        saving = (c_fp * (fp_s - fp_b) + c_fn * (fn_s - fn_b)) / n
        net = saving - premium
        per_concept[b] = net

    lo, hi = np.percentile(per_concept, [2.5, 97.5])
    return {
        "n_boot": n_boot,
        "per_concept": {"lo": float(lo), "hi": float(hi)},
    }


def bootstrap_auc_net_value(
    cb: dict, n_boot: int = COST_BENEFIT_BOOTSTRAP_N, seed: int = 0
) -> dict:
    """Cluster-bootstrap 95% CI for the threshold-free (AUC) net value.

    Same concept-level resampling as :func:`bootstrap_net_value`, but each
    resample is summarised by the area under its expected-cost curve (integrated
    over all thresholds) rather than a single ``t*`` value. Averaging over
    thresholds removes the volatility caused by flipping decisions at one hard
    cut-off, so this interval is expected to be narrower.
    """
    oof = cb["oof"]
    y = oof["y"]
    ps = oof["p_survey"]
    pb = oof["p_behavioural"]
    cidx = oof["concept_idx"]
    n_concepts = int(cidx.max()) + 1
    thresholds = cb["thresholds"]

    c_fp = cb["base"]["c_fp"]
    c_fn = cb["base"]["c_fn"]
    premium = cb["premium"]

    positions = [np.where(cidx == c)[0] for c in range(n_concepts)]
    rng = np.random.default_rng(seed)

    per_concept = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.integers(0, n_concepts, size=n_concepts)
        pos = np.concatenate([positions[c] for c in sample])
        yy = y[pos]
        auc_s = _auc_from_predictions(yy, ps[pos], thresholds, c_fp, c_fn)
        auc_b = _auc_from_predictions(yy, pb[pos], thresholds, c_fp, c_fn)
        per_concept[b] = (auc_s - auc_b) - premium

    lo, hi = np.percentile(per_concept, [2.5, 97.5])
    return {
        "n_boot": n_boot,
        "per_concept": {"lo": float(lo), "hi": float(hi)},
    }


def implicit_screen_analysis(df: pd.DataFrame) -> dict:
    """Cost-benefit of adding the implicit screen on top of the behavioural screen.

    Restricted to the 70 shared Combined-launched cases where ``Implicit_Score``
    exists, so the only difference between the two screens is the predictor set
    (behavioural = survey + behavioural measures; implicit = + ``Implicit_Score``).
    The premium is the Combined-vs-Behavioural research-cost delta. Returns the
    base-case FP/FN decomposition at the equal-cost ``t*``, the resulting net
    value, a C_FP x C_FN sensitivity grid, and the break-even false-negative
    cost — mirroring the survey-vs-behavioural cost-benefit treatment.
    """
    oof = cost_benefit_oof(
        df,
        base_predictors=BEHAVIOURAL_CV_PREDICTORS,
        full_predictors=IMPLICIT_CV_PREDICTORS,
    )
    y = oof["y"].astype(int)
    p_beh = oof["p_survey"]  # base screen
    p_imp = oof["p_behavioural"]  # full screen
    n = oof["n_oof"]

    sub = _shared_subset(df, IMPLICIT_CV_PREDICTORS)
    ls = pd.to_numeric(sub["Launch_Support_EUR"], errors="coerce").dropna()
    q25, median, q75 = ls.quantile(list(COST_BENEFIT_FP_QUANTILES)).to_numpy()

    rc = pd.to_numeric(df["Research_Cost_EUR"], errors="coerce")
    behavioural_cost = float(rc[df["Research_Package"] == "Behavioural"].mean())
    combined_cost = float(rc[df["Research_Package"] == "Combined"].mean())
    premium = combined_cost - behavioural_cost

    def net_value(c_fp: float, c_fn: float) -> dict:
        t_star = _optimal_threshold(c_fp, c_fn)

        def counts(p: np.ndarray) -> tuple[int, int, int]:
            pred = (p >= t_star).astype(int)
            fp = int(np.sum((pred == 1) & (y == 0)))
            fn = int(np.sum((pred == 0) & (y == 1)))
            launches = int(np.sum(pred == 1))
            return fp, fn, launches

        fp_b, fn_b, launches_b = counts(p_beh)
        fp_i, fn_i, launches_i = counts(p_imp)
        # decision saving (behavioural cost - implicit cost); positive means the
        # implicit screen's decisions are cheaper.
        decision_saving = (c_fp * (fp_b - fp_i) + c_fn * (fn_b - fn_i)) / n
        net = decision_saving - premium
        return {
            "c_fp": float(c_fp),
            "c_fn": float(c_fn),
            "t_star": float(t_star),
            "fp_behavioural": fp_b,
            "fn_behavioural": fn_b,
            "fp_implicit": fp_i,
            "fn_implicit": fn_i,
            "delta_fp": fp_i - fp_b,
            "delta_fn": fn_i - fn_b,
            "delta_launches": launches_i - launches_b,
            "decision_saving_per_concept": decision_saving,
            "net_per_concept": net,
        }

    base = net_value(median, median)

    fp_grid = [q25, median, q75]
    fn_grid = [q25, median, q75] + [m * median for m in COST_BENEFIT_FN_MULTIPLIERS]
    sensitivity_rows = []
    for c_fp in fp_grid:
        for c_fn in fn_grid:
            nv = net_value(c_fp, c_fn)
            sensitivity_rows.append(
                {
                    "C_FP": float(round(c_fp)),
                    "C_FN": float(round(c_fn)),
                    "t*": nv["t_star"],
                    "net_per_concept": nv["net_per_concept"],
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)

    # Break-even false-negative cost (C_FP held at the median), located by a
    # fine scan with linear interpolation over the plausible range.
    be_fn_scan = np.linspace(q25, 3.0 * median, 500)
    be_net = np.array([net_value(median, c)["net_per_concept"] for c in be_fn_scan])
    break_even: dict | None = None
    for j in range(len(be_net) - 1):
        if be_net[j] < 0 <= be_net[j + 1]:
            frac = (0.0 - be_net[j]) / (be_net[j + 1] - be_net[j])
            be_fn = float(be_fn_scan[j] + frac * (be_fn_scan[j + 1] - be_fn_scan[j]))
            break_even = {
                "c_fp": float(median),
                "c_fn": be_fn,
                "ratio": be_fn / median,
            }
            break

    return {
        "n_cases": oof["n_cases"],
        "n_oof": n,
        "base_rate": oof["base_rate"],
        "launch_support": {
            "q25": float(q25),
            "median": float(median),
            "q75": float(q75),
            "iqr": float(q75 - q25),
        },
        "premium": premium,
        "t_star": base["t_star"],
        "fp_behavioural": base["fp_behavioural"],
        "fn_behavioural": base["fn_behavioural"],
        "fp_implicit": base["fp_implicit"],
        "fn_implicit": base["fn_implicit"],
        "delta_fp": base["delta_fp"],
        "delta_fn": base["delta_fn"],
        "delta_launches": base["delta_launches"],
        "decision_saving_per_concept": base["decision_saving_per_concept"],
        "net_per_concept": base["net_per_concept"],
        "sensitivity": sensitivity,
        "break_even": break_even,
    }


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

        # 13. Logistic regression — classifying success -----------------------
        log.heading("13. Logistic regression — classifying success", level=2)
        log.paragraph(
            "Success is defined as ``Sales_vs_Target_Pct >= 100``. A regularised "
            "logistic regression (``LogisticRegression`` with an L2 penalty and "
            "balanced class weights) is fit to each of the three predictor sets, "
            "with predictors standardised to z-scores. Coefficients are reported "
            "per standard deviation; an odds ratio above 1 means a one-SD increase "
            "in the predictor raises the odds of success. All metrics are "
            "in-sample and therefore optimistic, and the sample-size, "
            "non-stationarity and selection caveats of sections 1–3 apply."
        )

        def _metric_rows(res: dict) -> list[list]:
            return [
                ["n", res["n"]],
                ["Success rate (%)", f"{res['success_rate']:.1f}"],
                ["Accuracy", f"{res['accuracy']:.3f}"],
                ["Balanced accuracy", f"{res['balanced_accuracy']:.3f}"],
                ["Precision", f"{res['precision']:.3f}"],
                ["Recall", f"{res['recall']:.3f}"],
                ["F1", f"{res['f1']:.3f}"],
                ["ROC-AUC", f"{res['roc_auc']:.3f}"],
            ]

        surv_log = logistic_model(df, SURVEY_MEASURES)
        beh_log = logistic_model(df, BEHAVIOURAL_CV_PREDICTORS)
        beh_base_log = logistic_model(
            df, SURVEY_MEASURES, complete_on=BEHAVIOURAL_CV_PREDICTORS
        )
        imp_log = logistic_model(df, IMPLICIT_CV_PREDICTORS)
        imp_base_log = logistic_model(
            df, BEHAVIOURAL_CV_PREDICTORS, complete_on=IMPLICIT_CV_PREDICTORS
        )

        log.heading("Survey measures", level=3)
        log.table_pair(
            surv_log["coefs"],
            _metric_rows(surv_log),
            right_headers=["Metric", "Value"],
            float_format="{:.3f}".format,
        )

        log.heading("Survey + behavioural", level=3)
        log.table_pair(
            beh_log["coefs"],
            _metric_rows(beh_log),
            right_headers=["Metric", "Value"],
            float_format="{:.3f}".format,
        )
        log.paragraph(
            f"Restricted to the same {beh_log['n']} cases, the survey-only model "
            f"reaches a ROC-AUC of {beh_base_log['roc_auc']:.3f}, so adding "
            f"``Behavioural_Choice_Pct`` lifts it to {beh_log['roc_auc']:.3f} — the "
            "behavioural measure is the strongest single predictor (odds ratio "
            f"{beh_log['coefs'].set_index('Term').loc['Behavioural_Choice_Pct', 'Odds ratio']:.2f})."
        )

        log.heading("Survey + behavioural + implicit", level=3)
        log.table_pair(
            imp_log["coefs"],
            _metric_rows(imp_log),
            right_headers=["Metric", "Value"],
            float_format="{:.3f}".format,
        )
        log.paragraph(
            f"Restricted to the same {imp_log['n']} cases, the behavioural model "
            f"reaches a ROC-AUC of {imp_base_log['roc_auc']:.3f}, so adding "
            f"``Implicit_Score`` lifts it only to {imp_log['roc_auc']:.3f}. As in the "
            "linear models, ``Purchase_Intent`` flips sign under collinearity on this "
            "small subset, and the implicit measure carries the largest odds ratio "
            f"({imp_log['coefs'].set_index('Term').loc['Implicit_Score', 'Odds ratio']:.2f})."
        )

        log.heading("Summary", level=3)
        summary_rows = []
        for label, res in [
            ("Survey", surv_log),
            ("Survey + behavioural", beh_log),
            ("Survey + behavioural + implicit", imp_log),
        ]:
            summary_rows.append(
                {
                    "Model": label,
                    "n": res["n"],
                    "Success rate (%)": f"{res['success_rate']:.1f}",
                    "Accuracy": f"{res['accuracy']:.3f}",
                    "Balanced accuracy": f"{res['balanced_accuracy']:.3f}",
                    "ROC-AUC": f"{res['roc_auc']:.3f}",
                }
            )
        log.table(pd.DataFrame(summary_rows))

        log.paragraph(
            "The behavioural measure produces the clearest gain: balanced accuracy "
            f"rises from {surv_log['balanced_accuracy']:.3f} (survey) to "
            f"{beh_log['balanced_accuracy']:.3f}, and ROC-AUC from "
            f"{surv_log['roc_auc']:.3f} to {beh_log['roc_auc']:.3f}. The implicit "
            "measure adds essentially nothing on the fair 70-case comparison "
            f"({imp_base_log['roc_auc']:.3f} → {imp_log['roc_auc']:.3f}), matching "
            "the regression result that its apparent in-sample benefit does not "
            "survive once the sample is restricted to the tiny Combined-launched "
            "subset. These conclusions are in-sample; out-of-sample validation "
            "(a time-ordered split, or repeated K-fold classification) is the "
            "natural next step and would carry the same caveats already recorded."
        )

        # 14. Time-ordered validation (2022–2024 → 2025) ----------------------
        log.heading("14. Time-ordered validation (2022–2024 → 2025)", level=2)
        log.paragraph(
            "The models are re-fit on 2022–2024 and evaluated on the held-out "
            "2025 concepts, so the test metrics are genuinely out of sample and "
            "match how the models would be used in practice (train on the past, "
            "predict the future). The 2025 concepts have a higher success rate "
            "than the training years, so the class distribution shifts between "
            "train and test — a further check on generalisation."
        )

        log_surv_tt = logistic_train_test(df, SURVEY_MEASURES)
        log_beh_tt = logistic_train_test(df, BEHAVIOURAL_CV_PREDICTORS)
        log_imp_tt = logistic_train_test(df, IMPLICIT_CV_PREDICTORS)

        summary_rows = []
        for label, res in [
            ("Survey", log_surv_tt),
            ("Survey + behavioural", log_beh_tt),
            ("Survey + behavioural + implicit", log_imp_tt),
        ]:
            summary_rows.append(
                {
                    "Model": label,
                    "n train": res["n_train"],
                    "n test": res["n_test"],
                    "Test success (%)": f"{res['test_success_rate']:.1f}",
                    "Train ROC-AUC": f"{res['train_roc_auc']:.3f}",
                    "Test ROC-AUC": f"{res['test_roc_auc']:.3f}",
                    "Test balanced accuracy": f"{res['test_balanced_accuracy']:.3f}",
                    "Test F1": f"{res['test_f1']:.3f}",
                    "Test Brier score": f"{res['test_brier']:.3f}",
                }
            )
        log.table(pd.DataFrame(summary_rows))

        log.paragraph(
            "The behavioural model is the only one that both fits and "
            "generalises: its test ROC-AUC ("
            f"{log_beh_tt['test_roc_auc']:.3f}) and balanced accuracy ("
            f"{log_beh_tt['test_balanced_accuracy']:.3f}) exceed the survey "
            f"model's ({log_surv_tt['test_roc_auc']:.3f} and "
            f"{log_surv_tt['test_balanced_accuracy']:.3f}), with a strong recall "
            f"of {log_beh_tt['test_recall']:.3f} and F1 of {log_beh_tt['test_f1']:.3f}."
        )
        log.paragraph(
            "The implicit model does not generalise: its in-sample fit is the "
            "best of the three (train ROC-AUC "
            f"{log_imp_tt['train_roc_auc']:.3f}) but it collapses out of sample to "
            f"a test ROC-AUC of {log_imp_tt['test_roc_auc']:.3f} and a balanced "
            f"accuracy of {log_imp_tt['test_balanced_accuracy']:.3f} — at or below "
            "chance — on only "
            f"{log_imp_tt['n_test']} test concepts. This mirrors the regression "
            "time-split (test R² −0.150) and confirms that the implicit model "
            "overfits the tiny training sample rather than capturing signal, "
            "exactly the risk flagged in section 3."
        )

        log.paragraph(
            "The precision-recall curves below show the same out-of-sample story "
            "in terms of precision (fraction of predicted successes that are "
            "correct) against recall (fraction of actual successes found), with "
            "the dashed line marking the no-skill baseline (the test-set success "
            "rate). Average precision (AP) summarises the area under each curve."
        )
        pr_fig_path = visualisation.plot_precision_recall(df)
        pr_fig_rel = os.path.relpath(pr_fig_path, log.path.parent).replace(os.sep, "/")
        log.image(
            "Precision-recall curves on the held-out 2025 concepts",
            pr_fig_rel,
        )

        log.paragraph(
            "The calibration curves below compare each model's predicted "
            "probability of success against the observed success frequency, with "
            "the diagonal marking perfect calibration. The Brier score (mean "
            "squared error of the predicted probability against the actual "
            "outcome, lower is better) is annotated on each panel; a no-skill "
            "model that always predicts the base rate would score about "
            "``p(1−p)``."
        )
        cal_fig_path = visualisation.plot_calibration(df)
        cal_fig_rel = os.path.relpath(cal_fig_path, log.path.parent).replace(os.sep, "/")
        log.image(
            "Calibration curves on the held-out 2025 concepts",
            cal_fig_rel,
        )

        # 15. Repeated K-fold CV — classification ----------------------------
        log.heading("15. Repeated K-fold CV — classification", level=2)
        log.paragraph(
            "The paired repeated K-fold design from sections 11–12 is repeated "
            "for classification, with the two models fit on identical folds of the "
            "shared subsample so the within-fold difference isolates the added "
            "predictor. The primary metric is ΔROC-AUC (higher is better), with "
            "Δbalanced accuracy and ΔF1 as supporting measures; a positive mean "
            "with a confidence interval mostly above zero indicates the added "
            "measure improves discrimination."
        )

        imp_cv_clf = repeated_kfold_compare_logistic(
            df, BEHAVIOURAL_CV_PREDICTORS, IMPLICIT_CV_PREDICTORS
        )
        beh_cv_clf = repeated_kfold_compare_logistic(
            df, SURVEY_MEASURES, BEHAVIOURAL_CV_PREDICTORS
        )

        log.heading("Implicit vs behavioural", level=3)
        log.table(classification_diff_summary_table(imp_cv_clf, "Implicit"))
        log.paragraph(
            "Adding ``Implicit_Score`` over the behavioural model changes nothing "
            "on average: mean ΔROC-AUC is essentially zero and the implicit model "
            "wins on only ~half of folds, while Δbalanced accuracy and ΔF1 are "
            "slightly negative (implicit better on ~33% of folds)."
        )

        log.heading("Behavioural vs survey", level=3)
        log.table(classification_diff_summary_table(beh_cv_clf, "Behavioural"))
        log.paragraph(
            "Adding ``Behavioural_Choice_Pct`` over the survey model gives a "
            "consistent, positive effect: mean ΔROC-AUC of +0.059, with the "
            "behavioural model better on ~85% of folds. This mirrors the "
            "regression result and confirms the behavioural measure's "
            "classification value is robust, whereas the implicit measure's is "
            "not — consistent with the tiny Combined-launched subsample and the "
            "pilot-study recommendation already recorded."
        )

        clf_fig_path = visualisation.plot_classification_cv_diffs(df)
        clf_fig_rel = os.path.relpath(clf_fig_path, log.path.parent).replace(os.sep, "/")
        log.image(
            "Repeated K-fold CV (classification): distribution of fold differences",
            clf_fig_rel,
        )

        # 16. Propensity score / inverse probability weighting -----------------
        log.heading("16. Propensity score / inverse probability weighting", level=2)
        log.paragraph(
            "Package selection is confounded: teams chose Behavioural/Combined "
            "packages non-randomly (likely according to prior confidence), so the "
            "raw success gap may partly reflect that selection rather than the "
            "package's own value. The unmeasured confidence confounder cannot be "
            "adjusted for, but the measured confounders can. A regularised "
            "logistic regression predicts the rich package (Behavioural or "
            "Combined vs Survey) from ``Test_Year``, ``Category`` and "
            "``Innovation_Type``. ``Sample_Size`` is excluded (negligibly "
            "correlated with outcomes), as are ``Research_Cost_EUR`` and "
            "``Turnaround_Days`` (consequences of the package choice, not "
            "confounders)."
        )
        ipw = propensity_ipw(df)
        log.heading("Propensity model", level=3)
        log.table(ipw["coefs"], float_format="{:.3f}".format)

        log.heading("Propensity-score distributions (before weighting)", level=3)
        log.table(ipw["ps_summary"], float_format="{:.3f}".format)
        log.paragraph(
            f"The propensity model is fit on all {ipw['n_full']} concepts; the "
            f"effect is estimated on the {ipw['n_outcome']} launched concepts with "
            "a recorded target. Stabilised inverse-propensity weights are trimmed "
            f"at the 1st/99th percentile ({(ipw['n_trimmed'])} weights trimmed)."
        )

        log.heading("Covariate balance (standardised mean differences)", level=3)
        log.table(ipw["balance"], float_format="{:.3f}".format)
        smd_before = float(ipw["balance"]["SMD before"].abs().max())
        smd_after = float(ipw["balance"]["SMD after"].abs().max())
        log.paragraph(
            "Before weighting, the ``Test_Year`` dummies are strongly imbalanced "
            "(as expected from the package-mix shift in section 2). After inverse "
            "propensity weighting the largest absolute standardised mean "
            f"difference falls from {smd_before:.3f} to {smd_after:.3f}, "
            "indicating the measured confounders are largely balanced."
        )

        boot = bootstrap_ipw_ate(df)
        raw = ipw["unweighted"]
        wtd = ipw["weighted"]
        effect = pd.DataFrame(
            [
                {
                    "Estimate": "Survey success (%)",
                    "Unweighted": f"{raw['survey'] * 100:.1f}",
                    "Weighted": f"{wtd['survey'] * 100:.1f}",
                },
                {
                    "Estimate": "Rich-package success (%)",
                    "Unweighted": f"{raw['rich'] * 100:.1f}",
                    "Weighted": f"{wtd['rich'] * 100:.1f}",
                },
                {
                    "Estimate": "Difference (ATE, pp)",
                    "Unweighted": f"{raw['ate'] * 100:+.1f}",
                    "Weighted": f"{wtd['ate'] * 100:+.1f}",
                },
            ]
        )
        log.heading("Effect estimate", level=3)
        log.table(effect)
        log.paragraph(
            "Unweighted, the rich package is associated with a "
            f"{raw['ate'] * 100:+.1f} pp higher success rate than Survey. After "
            "inverse propensity weighting the estimate is "
            f"{wtd['ate'] * 100:+.1f} pp (bootstrap 95% CI "
            f"{boot['lo'] * 100:+.1f} to {boot['hi'] * 100:+.1f} pp, "
            f"{boot['n_boot']} resamples)."
        )
        log.paragraph(
            "Adjusting for the measured confounders narrows the gap relative to "
            "the raw comparison, confirming that part of the raw behavioural/"
            "implicit advantage reflects the packages being concentrated in "
            "later, more favourable cohorts. The residual effect remains positive "
            "but is observational: the dominant confounder — the team's "
            "unmeasured prior confidence in a concept — is not in the data, so "
            "the weighted estimate is a robustness check on the package "
            "advantage rather than a causal effect."
        )

        ipw_fig_path = visualisation.plot_propensity_overlap(df)
        ipw_fig_rel = os.path.relpath(ipw_fig_path, log.path.parent).replace(os.sep, "/")
        log.image(
            "Propensity-score overlap before and after inverse probability weighting",
            ipw_fig_rel,
        )

        # 17. Cost-benefit analysis -------------------------------------------
        log.heading(
            "17. Cost-benefit analysis — is the behavioural package worth its added cost?",
            level=2,
        )
        log.paragraph(
            "The behavioural screen predicts launch success better than the survey "
            "screen (sections 11-15). This section asks whether that extra "
            "predictive value is worth the behavioural package's added research "
            "cost. The decision is a go/no-go launch call made from a predicted "
            "probability of success (``Sales_vs_Target_Pct >= 100``); the baseline "
            "to beat is the survey screen making the same call on the same concepts."
        )

        cb = cost_benefit_analysis(df)
        imp_screen = implicit_screen_analysis(df)
        ls = cb["launch_support"]

        log.heading("Cost model", level=3)
        log.paragraph(
            "Each concept has four possible outcomes. A true positive (launch of a "
            "winner) and a true negative (stop of a loser) cost nothing. A false "
            "positive (launch of a loser) costs the median ``Launch_Support_EUR``. "
            "A false negative (stop of a winner) costs the missed profit, which is "
            "not an observation in the data, so on the first pass it is set equal "
            "to the false-positive cost. The behavioural package's added cost "
            "relative to Survey is the exploration-log delta."
        )
        log.key_values(
            [
                ["C_FP — launch a loser", f"€{ls['median']:,.0f}"],
                ["C_FN — stop a winner (first pass)", f"€{ls['median']:,.0f}"],
                ["Launch_Support_EUR IQR", f"€{ls['q25']:,.0f} - €{ls['q75']:,.0f}"],
                ["Behavioural premium vs Survey", f"€{cb['premium']:,.0f} / concept"],
            ]
        )

        log.heading("Evaluation set, folds and calibration", level=3)
        log.paragraph(
            f"The evaluation set is the {cb['n_cases']} launched concepts with "
            "behavioural measures (Behavioural + Combined packages). Both screens "
            "are fit on identical folds of the repeated K-fold design "
            f"({cb['n_folds']} folds, {cb['n_oof']} out-of-fold predictions) at a "
            f"success base rate of {cb['base_rate']*100:.1f}%."
        )
        cal = cb["calibration"]
        base_brier = cb["base_rate"] * (1 - cb["base_rate"])
        cal_rows = [
            {
                "Screen": "Survey",
                "Raw Brier": f"{cal['brier_survey_raw']:.3f}",
                "Recalibrated Brier": f"{cal['brier_survey']:.3f}",
                "Base-rate Brier": f"{base_brier:.3f}",
            },
            {
                "Screen": "Behavioural",
                "Raw Brier": f"{cal['brier_behavioural_raw']:.3f}",
                "Recalibrated Brier": f"{cal['brier_behavioural']:.3f}",
                "Base-rate Brier": f"{base_brier:.3f}",
            },
        ]
        log.table(pd.DataFrame(cal_rows))
        log.paragraph(
            "The models use balanced class weights, which shift the intercept and "
            "can bias raw probabilities toward 0.5, so each fold's probabilities "
            "are Platt-recalibrated (a sigmoid fit on the training probabilities) "
            "before thresholding. Here the base rate is close to 50%, so the "
            "recalibration changes the probabilities only marginally (Brier scores "
            "above are essentially unchanged); the sweep below still uses the "
            "recalibrated probabilities, a precondition for the analytical "
            "threshold to be optimal."
        )
        log.heading("Confusion-matrix sweep", level=3)
        log.paragraph(
            "For every threshold ``t`` in [0, 1] (steps of "
            f"{COST_BENEFIT_THRESHOLD_STEP:.2f}) the recalibrated probabilities are "
            "thresholded into launch/stop decisions and the TP/TN/FP/FN counts are "
            "tallied across all held-out folds, separately for each screen."
        )
        log.heading("Survey screen", level=4)
        log.table(cb["sweep"]["survey"], float_format="{:.2f}".format)
        log.heading("Behavioural screen", level=4)
        log.table(cb["sweep"]["behavioural"], float_format="{:.2f}".format)

        cb_fig_path = visualisation.plot_cost_benefit_ev(result=cb)
        cb_fig_rel = os.path.relpath(cb_fig_path, log.path.parent).replace(os.sep, "/")
        log.heading("Expected cost against threshold", level=3)
        log.paragraph(
            "Each confusion matrix is converted to an expected cost per concept by "
            "applying the FP and FN costs to the corresponding cells (TP and TN "
            "cost nothing) and summing. The curve is plotted below for both "
            "screens; lower cost is better, so the cost-minimising threshold is "
            "the optimum (equivalently, the value-maximising threshold)."
        )
        log.image("Expected cost per concept against the decision threshold", cb_fig_rel)

        b = cb["base"]
        log.heading("Cost-minimising threshold", level=3)
        log.paragraph(
            "For a calibrated model with correct decisions free of charge, the "
            "cost-minimising threshold is ``t* = C_FP / (C_FP + C_FN)``, which is "
            f"{b['t_star']:.3f} when C_FP = C_FN. The empirical minima of the sweep "
            f"sit at t = {b['empirical_argmin_survey']:.2f} (survey) and "
            f"t = {b['empirical_argmin_behavioural']:.2f} (behavioural). The survey "
            "minimum is close to t*, but the behavioural minimum is higher, a sign "
            "its probabilities are not perfectly calibrated and that the headline "
            "net value at t* is conservative — the behavioural screen's saving "
            "peaks at a higher threshold."
        )

        log.heading("Net value of switching from survey to behavioural", level=3)
        log.key_values(
            [
                ["Optimal threshold t*", f"{b['t_star']:.3f}"],
                ["Survey expected cost / concept", f"€{b['cost_survey_at_tstar']:,.0f}"],
                ["Behavioural expected cost / concept", f"€{b['cost_behavioural_at_tstar']:,.0f}"],
                ["Decision saving / concept (survey - behavioural)", f"€{b['decision_saving_per_concept']:+,.0f}"],
                ["Net value / concept (saving - premium)", f"€{b['net_per_concept']:+,.0f}"],
            ]
        )
        log.paragraph(
            f"The €{b['decision_saving_per_concept']:+,.0f} per-concept saving comes from "
            f"{b['error_reduction']} fewer errors across the {cb['n_oof']:,} held-out "
            "decisions, but that net figure hides a lop-sided composition. Relative to "
            "the survey screen at t*, the behavioural screen makes "
            f"{b['delta_fp']:+d} more false positives (launches of losers) while "
            f"making {b['delta_fn']:+d} fewer false negatives (stops of winners): "
            f"it launches {b['launches_behavioural'] - b['launches_survey']:,} more "
            "concepts overall, i.e. it operates more aggressively. Its entire error "
            "advantage is on the false-negative side, and that side is precisely where "
            "the two costs cancel at C_FP = C_FN; only when C_FN exceeds C_FP does the "
            "saving outweigh the extra false positives. This is why the net value "
            "hinges so directly on the assumed cost of a stopped winner."
        )
        log.table(
            [
                ["False positive (launch a loser)", f"{b['fp_survey']:,}", f"{b['fp_behavioural']:,}", f"{b['delta_fp']:+,}"],
                ["False negative (stop a winner)", f"{b['fn_survey']:,}", f"{b['fn_behavioural']:,}", f"{b['delta_fn']:+,}"],
                ["Total errors", f"{b['fp_survey'] + b['fn_survey']:,}", f"{b['fp_behavioural'] + b['fn_behavioural']:,}", f"{-b['error_reduction']:+,}"],
            ],
            headers=["Error type", "Survey", "Behavioural", "Δ (behavioural − survey)"],
        )
        lo = cb["iqr_range"]["low"]
        hi = cb["iqr_range"]["high"]
        log.paragraph(
            "Because ``Launch_Support_EUR`` is right-skewed, the net value is also "
            "reported across its interquartile range (C_FP = C_FN at Q1 and Q3). "
            f"At Q1 (€{lo['c_fp']:,.0f}) the net value is "
            f"€{lo['net_per_concept']:+,.0f} per concept; at Q3 "
            f"(€{hi['c_fp']:,.0f}) it is €{hi['net_per_concept']:+,.0f} per concept."
        )

        pl = cb["plausible"]
        log.heading("Plausible scenario — C_FN = 1.5 × C_FP (profit, not break-even)", level=3)
        log.paragraph(
            "The equal-cost first pass is a break-even model. In practice a "
            "launched winner's profit margin exceeds the launch spend, so the "
            "missed profit of a stopped winner (C_FN) should exceed the cost of a "
            "failed launch (C_FP). Setting C_FN = 1.5 × C_FP (here "
            f"€{pl['c_fp']:,.0f} vs €{pl['c_fn']:,.0f}) lowers the optimal threshold "
            f"to t* = {pl['t_star']:.2f} — launch more aggressively, because missing "
            "a winner now costs more than launching a loser."
        )
        log.key_values(
            [
                ["C_FP (launch a loser)", f"€{pl['c_fp']:,.0f}"],
                ["C_FN (stop a winner)", f"€{pl['c_fn']:,.0f}"],
                ["Optimal threshold t*", f"{pl['t_star']:.2f}"],
                ["Survey expected cost / concept", f"€{pl['cost_survey_at_tstar']:,.0f}"],
                ["Behavioural expected cost / concept", f"€{pl['cost_behavioural_at_tstar']:,.0f}"],
                ["Decision saving / concept (survey - behavioural)", f"€{pl['decision_saving_per_concept']:+,.0f}"],
                ["Net value / concept (saving - premium)", f"€{pl['net_per_concept']:+,.0f}"],
            ]
        )

        eb = cb["base"]
        log.heading("Each screen at its own empirical threshold", level=3)
        log.paragraph(
            "The theoretical break-even threshold t* = 0.5 assumes both models "
            "are calibrated. As a robustness check the net value is also computed "
            "when each screen operates at its own empirical cost-minimising "
            f"threshold (t = {eb['empirical_argmin_survey']:.2f} for survey, "
            f"t = {eb['empirical_argmin_behavioural']:.2f} for behavioural). This "
            "gives each screen the benefit of its best observed operating point "
            "on the held-out folds, so the saving is optimistic relative to t*."
        )
        log.key_values(
            [
                ["Survey threshold", f"t = {eb['empirical_argmin_survey']:.2f}"],
                ["Survey cost at that threshold", f"€{eb['empirical_cost_survey']:,.0f}"],
                ["Behavioural threshold", f"t = {eb['empirical_argmin_behavioural']:.2f}"],
                ["Behavioural cost at that threshold", f"€{eb['empirical_cost_behavioural']:,.0f}"],
                ["Decision saving / concept (survey - behavioural)", f"€{eb['empirical_decision_saving_per_concept']:+,.0f}"],
                ["Net value / concept (saving - premium)", f"€{eb['empirical_net_per_concept']:+,.0f}"],
            ]
        )

        boot = bootstrap_net_value(cb)
        log.heading("Bootstrap confidence interval", level=3)
        log.paragraph(
            f"To attach uncertainty to the headline net value, a cluster "
            f"bootstrap resamples the {cb['n_cases']} concepts (with replacement) "
            "and recomputes the net value at t* from the resampled out-of-fold "
            f"predictions ({boot['n_boot']:,} resamples). Resampling is done at "
            "the concept level so the within-concept correlation across the 20 CV "
            "repeats is preserved; the premium is a recorded cost and is held "
            "fixed. Because it reweights the fixed out-of-fold predictions rather "
            "than refitting, this interval reflects concept-sampling uncertainty "
            "only and should be read as a lower bound on the true uncertainty."
        )
        if boot["per_concept"]["hi"] < 0:
            ci_note = (
                "The interval lies entirely below zero, so the conclusion that "
                "the package does not pay for itself at equal costs is robust to "
                "concept-sampling uncertainty."
            )
        elif boot["per_concept"]["lo"] > 0:
            ci_note = (
                "The interval lies entirely above zero, so the package would be "
                "robustly worth its premium at equal costs."
            )
        else:
            ci_note = (
                "The interval straddles zero, so the sign of the net value is not "
                "robust at the current sample size."
            )
        log.table(
            pd.DataFrame(
                [
                    {
                        "Net value": "Per concept",
                        "Point estimate (€)": f"{b['net_per_concept']:+,.0f}",
                        "95% CI (€)": f"[{boot['per_concept']['lo']:+,.0f}, {boot['per_concept']['hi']:+,.0f}]",
                    },
                ]
            )
        )
        log.paragraph(ci_note)

        boot_auc = bootstrap_auc_net_value(cb)
        log.heading("Threshold-free summary (area under the cost curve)", level=3)
        log.paragraph(
            "The net value at t* depends on a single hard threshold, which is "
            "volatile because a concept sitting just either side of t* flips its "
            "whole decision. As a threshold-independent alternative, the expected "
            "cost curve is integrated over the full [0, 1] threshold range (area "
            "under the curve, lower is better), so every threshold contributes "
            "rather than one. This averages out that single-cut-off volatility."
        )
        auc = cb["auc"]
        log.key_values(
            [
                ["Survey AUC (€ / concept)", f"{auc['auc_survey']:,.0f}"],
                ["Behavioural AUC (€ / concept)", f"{auc['auc_behavioural']:,.0f}"],
                ["Threshold-free saving / concept", f"€{auc['decision_saving_per_concept']:+,.0f}"],
                ["Threshold-free net value / concept", f"€{auc['net_per_concept']:+,.0f}"],
            ]
        )
        tstar_width = boot["per_concept"]["hi"] - boot["per_concept"]["lo"]
        auc_width = boot_auc["per_concept"]["hi"] - boot_auc["per_concept"]["lo"]
        ratio = tstar_width / auc_width if auc_width > 0 else float("inf")
        log.paragraph(
            f"The cluster bootstrap gives a 95% CI of "
            f"[€{boot_auc['per_concept']['lo']:+,.0f}, €{boot_auc['per_concept']['hi']:+,.0f}] "
            f"for the threshold-free net value, a width of €{auc_width:,.0f} versus "
            f"€{tstar_width:,.0f} for the t* figure — a {ratio:.1f}× reduction. "
            "Averaging over thresholds therefore reduces variance, as expected, "
            "though the interval still straddles zero so the sign of the net value "
            "remains uncertain at this sample size."
        )

        fn_mult_text = ", ".join(f"{m:g}×" for m in COST_BENEFIT_FN_MULTIPLIERS)
        log.heading("Sensitivity to the cost of a false negative", level=3)
        log.paragraph(
            "The missed-profit cost of a false negative is unknown, so the net "
            "value is recomputed with C_FN at the Q1/median/Q3 values plus "
            f"{fn_mult_text} the median launch spend, and C_FP at the "
            "Q1/median/Q3 values, each cell evaluated at its own ``t*``. The "
            "grid is not monotone: because the behavioural screen's out-of-sample "
            "advantage is modest (sections 11-15), the net value is negative "
            "across most of the grid and positive only in a narrow band where "
            "C_FN modestly exceeds C_FP, collapsing at the extremes where both "
            "screens converge on the same decision."
        )
        sens_rows = [
            {
                "C_FP (€)": f"{row['C_FP']:,.0f}",
                "C_FN (€)": f"{row['C_FN']:,.0f}",
                "t*": f"{row['t*']:.3f}",
                "Net value / concept (€)": f"{row['net_per_concept']:+,.0f}",
            }
            for row in cb["sensitivity"].to_dict("records")
        ]
        log.table(pd.DataFrame(sens_rows))

        cb_sens_path = visualisation.plot_cost_benefit_sensitivity(result=cb)
        cb_sens_rel = os.path.relpath(cb_sens_path, log.path.parent).replace(os.sep, "/")
        log.image("Net value per concept over the FP/FN cost grid", cb_sens_rel)

        be = cb["break_even"]
        if be is not None:
            log.heading("Break-even false-negative cost", level=3)
            log.paragraph(
                "At the median launch spend (C_FP = "
                f"€{be['c_fp']:,.0f}), the net value crosses from negative to "
                "positive when the false-negative cost reaches approximately "
                f"**C_FN = €{be['c_fn']:,.0f}** — that is "
                f"**{be['ratio']:.2f}× the launch spend**. This is the single "
                "decision number: the behavioural package pays for its premium "
                "if and only if a stopped winner's forgone profit exceeds this "
                "threshold. At higher C_FN the net value turns negative again "
                "as both screens converge on the same decision (the grid above "
                "is non-monotone)."
            )

        log.heading("Adding the implicit screen", level=3)
        log.paragraph(
            "The implicit measure is only collected for Combined packages, so a "
            "three-predictor \"implicit screen\" (survey + behavioural + "
            "``Implicit_Score``) can only be run on the 70 launched Combined "
            "concepts where the implicit score exists. To isolate the implicit "
            "measure's contribution, both screens are fit on identical folds of "
            "that shared subset, so the only difference is the predictor set, and "
            "the premium is the Combined-vs-Behavioural research-cost delta "
            f"(€{imp_screen['premium']:,.0f} per concept)."
        )
        log.paragraph(
            "The implicit screen does **not** reduce false positives. At the "
            f"equal-cost t* = {imp_screen['t_star']:.1f} it makes "
            f"{imp_screen['delta_fp']:+d} more false positives and only "
            f"{imp_screen['delta_fn']:+d} fewer false negatives than the "
            "behavioural screen — i.e. it launches even more aggressively, and its "
            "whole (tiny) gain is again on the false-negative side. The net effect "
            f"is {imp_screen['delta_fp'] + imp_screen['delta_fn']:+d} more errors "
            f"across the {imp_screen['n_oof']:,} held-out decisions, so its decision "
            f"cost is €{-imp_screen['decision_saving_per_concept']:,.0f} per concept "
            "higher than the behavioural screen before any premium. Adding the "
            f"€{imp_screen['premium']:,.0f} premium, the net value of the implicit "
            f"screen is €{imp_screen['net_per_concept']:+,.0f} per concept — "
            "negative, on top of the behavioural screen already failing to clear "
            "its own premium at equal costs."
        )
        log.table(
            [
                ["False positive (launch a loser)", f"{imp_screen['fp_behavioural']:,}", f"{imp_screen['fp_implicit']:,}", f"{imp_screen['delta_fp']:+,}"],
                ["False negative (stop a winner)", f"{imp_screen['fn_behavioural']:,}", f"{imp_screen['fn_implicit']:,}", f"{imp_screen['delta_fn']:+,}"],
                ["Total errors", f"{imp_screen['fp_behavioural'] + imp_screen['fn_behavioural']:,}", f"{imp_screen['fp_implicit'] + imp_screen['fn_implicit']:,}", f"{imp_screen['delta_fp'] + imp_screen['delta_fn']:+,}"],
            ],
            headers=["Error type", "Behavioural", "Implicit", "Δ (implicit − behavioural)"],
        )
        log.paragraph(
            "This is consistent with the earlier classification CV (section 15), "
            "where the implicit screen was a coin-flip over the behavioural screen "
            "on the same 70 cases (Δ ROC-AUC +0.010, better on only 51% of folds). "
            "The behavioural measure is the point of diminishing returns: the "
            "implicit measure adds research cost and no decision value, so the "
            "answer to whether it could rescue the net value by cutting false "
            "positives is no — it pushes the error mix the wrong way. If an "
            "implicit measure is ever to justify its cost, it would have to come "
            "from a much larger sample (the 70-case subset is far too small to "
            "resolve its contribution), which is again a pilot-study question."
        )

        log.heading("Implicit screen — sensitivity to FP/FN costs", level=4)
        log.paragraph(
            "Mirroring the survey-vs-behavioural treatment, the implicit screen's "
            "net value is recomputed over the same C_FP × C_FN grid (C_FP at "
            f"Q1/median/Q3 of the 70-case launch support; C_FN at Q1/median/Q3 "
            f"plus {fn_mult_text} the median), each cell at its own ``t*``."
        )
        imp_sens_rows = [
            {
                "C_FP (€)": f"{row['C_FP']:,.0f}",
                "C_FN (€)": f"{row['C_FN']:,.0f}",
                "t*": f"{row['t*']:.3f}",
                "Net value / concept (€)": f"{row['net_per_concept']:+,.0f}",
            }
            for row in imp_screen["sensitivity"].to_dict("records")
        ]
        log.table(pd.DataFrame(imp_sens_rows))

        imp_sens_path = visualisation.plot_cost_benefit_sensitivity(
            sensitivity=imp_screen["sensitivity"],
            title="Implicit screen: net value per concept (EUR) vs FP/FN costs",
            filename="cost_benefit_sensitivity_implicit.png",
        )
        imp_sens_rel = os.path.relpath(imp_sens_path, log.path.parent).replace(os.sep, "/")
        log.image("Implicit-screen net value per concept over the FP/FN cost grid", imp_sens_rel)

        imp_be = imp_screen["break_even"]
        if imp_be is not None:
            log.paragraph(
                "At the median launch spend (C_FP = "
                f"€{imp_be['c_fp']:,.0f}), the implicit screen's net value would "
                "cross zero only at a false-negative cost of "
                f"**C_FN = €{imp_be['c_fn']:,.0f}** — "
                f"**{imp_be['ratio']:.2f}× the launch spend**."
            )
        else:
            log.paragraph(
                "Unlike the behavioural screen, the implicit screen has **no "
                "break-even point** within the plausible range: its net value is "
                "negative across the entire grid (C_FN up to 3× the median launch "
                "spend). The reason is structural — its decision saving is itself "
                "negative at equal costs (more false positives and hardly fewer "
                "false negatives), so no realistic false-negative cost can offset "
                "both that and the €3,112 premium. Only at an implausibly large "
                "C_FN (well beyond 3× the launch spend) would its net value turn "
                "positive."
            )

        if b["net_per_concept"] >= 0:
            verdict = (
                f"the behavioural package **pays for its premium** at equal costs "
                f"(net value €{b['net_per_concept']:+,.0f} per concept)."
            )
        else:
            verdict = (
                "the behavioural package does **not yet pay for its premium** at "
                f"equal costs (net value €{b['net_per_concept']:+,.0f} per concept)."
            )
        log.paragraph(
            f"**Interpretation.** At the first-pass cost model, {verdict} Its "
            f"improved decisions save €{b['decision_saving_per_concept']:,.0f} per "
            f"concept, which is less than the €{cb['premium']:,.0f} premium. The "
            "saving is real but modest — consistent with sections 11-15, where the "
            "behavioural advantage was positive yet not statistically significant "
            "under resampling — and it exceeds the premium only in a narrow band of "
            "the cost grid. The dominant unknown is therefore the profit forgone by "
            "stopping a winner (C_FN), not the research premium; a pilot study "
            "should quantify that profit directly, because it, not the premium, "
            "decides whether the behavioural package is worth its added cost. "
            "Evaluated at each screen's own best observed threshold instead, the "
            f"net value is €{eb['empirical_net_per_concept']:+,.0f} per concept. "
            "Under the plausible profit-margin scenario (C_FN = 1.5 x C_FP), "
            "behavioural does pay for itself — "
            f"€{pl['net_per_concept']:+,.0f} per concept — though only marginally, "
            "and this positive figure inherits the same wide uncertainty as the "
            "break-even estimate."
        )

    print(f"Modelling log written to: {log.path}")


if __name__ == "__main__":
    main()

