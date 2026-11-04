"""Hyperparameter search for lab 2's gradient boosting: grid, random and Bayesian.

The tuned pipeline is lab 2's scenario B, full cleaning with target encoding of
the high-cardinality columns, ending in histogram gradient boosting: the best
ranking model there, and the one with the most hyperparameters. Every search
scores a candidate with the same stratified 3-fold cross-validation, which is
also the inner loop of the nested scheme in mk1.nested, so the best score of a
search here and an inner score there are measured the same way.

No search fits a final model (refit=False throughout): the time of a search is
the time it takes to choose the hyperparameters, and the caller that needs the
chosen model fits it with build_tuned().

Method labels and figure text stay in Ukrainian: they are report content.
"""

import logging
import time
from dataclasses import asdict, dataclass
from math import prod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from matplotlib.colors import Normalize
from optuna import create_study
from optuna.logging import WARNING as OPTUNA_WARNING
from optuna.logging import disable_default_handler, enable_propagation, set_verbosity
from optuna.samplers import TPESampler
from optuna.trial import Trial
from scipy.stats import loguniform
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.utils import check_random_state

from mk1.artifacts import save_figure, save_json
from mk1.config import CV_FOLDS, INNER_FOLDS, LAB3, SEED
from mk1.scenarios import SCENARIO_BY_KEY, build_scenario

log = logging.getLogger(__name__)

# Both metrics are recorded for every candidate; PR-AUC alone selects. It is the
# main metric of labs 1 and 2: with one positive in eleven, ROC-AUC rewards the
# easy ranking of the many negatives and moves little between models.
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}
SELECT = "pr_auc"

# Every core. Under joblib's process backend each worker is limited to
# cpu_count // n_jobs threads, so the boosting's own OpenMP threads do not
# oversubscribe the machine.
N_JOBS = -1

# The full grid: three values per hyperparameter, 3^4 = 81 candidates. Each list
# holds an interval's two bounds and their geometric middle; random and Bayesian
# search draw log-uniformly from the same intervals, so all three explore the
# same region and differ only in how they choose points inside it.
GRID: dict[str, list[float]] = {
    "learning_rate": [0.02, 0.063, 0.2],
    "max_leaf_nodes": [8, 32, 128],
    "min_samples_leaf": [20, 100, 500],
    "l2_regularization": [0.1, 1.0, 10.0],
}
INTEGER = frozenset({"max_leaf_nodes", "min_samples_leaf"})

# Trials that Optuna's TPE draws at random before it models the results. The
# default of 10 would spend half of a 20-trial budget on random points, which
# leaves the Bayesian part of the comparison too little to show.
STARTUP_TRIALS = 5

BASELINE_JSON = LAB3 / "baseline.json"
TUNING_JSON = LAB3 / "tuning.json"


@dataclass(frozen=True)
class Method:
    """
    One hyperparameter search strategy.

    Args:
        key: Identifier used in the metric files.
        label: Name shown in the report figures.
        kind: "grid", "random" or "bayes".
        n_iter: Candidates to evaluate; None for the full grid.
        color: Colour of the method in every figure.
    """

    key: str
    label: str
    kind: str
    n_iter: int | None
    color: str


GRID_SIZE = prod(len(values) for values in GRID.values())

METHODS = (
    Method("grid", f"Grid Search, N = {GRID_SIZE}", "grid", None, "dimgrey"),
    Method("random-20", "Random Search, N = 20", "random", 20, "lightskyblue"),
    Method("random-50", "Random Search, N = 50", "random", 50, "steelblue"),
    Method("bayes-20", "Bayesian (TPE), N = 20", "bayes", 20, "darkorange"),
)
METHOD_BY_KEY = {m.key: m for m in METHODS}


@dataclass(frozen=True)
class SearchResult:
    """
    Outcome of one search.

    Args:
        best_params: Hyperparameters of the candidate with the highest PR-AUC.
        best_score: Its mean PR-AUC over the search's folds.
        best_std: Standard deviation of that PR-AUC over the folds.
        best_roc_auc: Its mean ROC-AUC.
        wall_seconds: Elapsed time of the whole search.
        compute_seconds: Fit and scoring time summed over every fold of every
            candidate; unlike the wall time, it does not depend on how many
            fits ran in parallel.
        history: Every candidate in the order it was proposed, with its
            parameters, mean scores and compute seconds.
    """

    best_params: dict[str, float]
    best_score: float
    best_std: float
    best_roc_auc: float
    wall_seconds: float
    compute_seconds: float
    history: list[dict]


class LogUniformInt:
    """
    Integers uniform on the log scale, in the form ParameterSampler accepts.

    scipy's randint is uniform on the linear scale, which would put most of the
    leaf-count draws above 64 — the slowest and least regularised trees — and
    differ from the log-scaled integers Optuna draws for the same parameter.

    Args:
        low: Smallest value, inclusive.
        high: Largest value, inclusive.
    """

    def __init__(self, low: int, high: int) -> None:
        self.low = low
        self.high = high

    def rvs(self, random_state: int | np.random.RandomState | None = None) -> int:
        """Draw one value."""
        rng = check_random_state(random_state)
        value = np.exp(rng.uniform(np.log(self.low), np.log(self.high)))
        return int(np.clip(np.round(value), self.low, self.high))


def build_tuned(params: dict[str, float]) -> Pipeline:
    """
    Lab 2's scenario B pipeline with the boosting's hyperparameters overridden.

    Args:
        params: Values for some or all of the GRID hyperparameters; the
            missing ones keep lab 2's values.

    Returns:
        An unfitted pipeline.
    """
    pipe = build_scenario(SCENARIO_BY_KEY["B"], "hgb")
    return pipe.set_params(
        **{f"model__{name}": value for name, value in params.items()}
    )


def baseline_params() -> dict[str, float]:
    """Lab 2's values of the tuned hyperparameters."""
    model = build_tuned({}).named_steps["model"]
    return {name: getattr(model, name) for name in GRID}


def inner_folds() -> StratifiedKFold:
    """The folds of every search and of the inner loop of nested CV."""
    return StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)


def outer_folds() -> StratifiedKFold:
    """The folds of the outer loop of nested CV, and of every unbiased estimate."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)


def distributions() -> dict[str, object]:
    """Random-search distributions over the bounds of each GRID interval."""
    out: dict[str, object] = {}
    for name, values in GRID.items():
        low, high = min(values), max(values)
        out[f"model__{name}"] = (
            LogUniformInt(int(low), int(high))
            if name in INTEGER
            else loguniform(low, high)
        )
    return out


def suggest(trial: Trial, name: str) -> float:
    """Ask Optuna for one hyperparameter, log-scaled over its GRID interval."""
    low, high = min(GRID[name]), max(GRID[name])
    if name in INTEGER:
        return trial.suggest_int(name, int(low), int(high), log=True)
    return trial.suggest_float(name, low, high, log=True)


def candidate(params: dict[str, float], scores: dict[str, np.ndarray]) -> dict:
    """One history entry from the per-fold output of cross_validate()."""
    return {
        "params": params,
        "pr_auc": float(np.mean(scores["test_pr_auc"])),
        "pr_auc_std": float(np.std(scores["test_pr_auc"], ddof=1)),
        "roc_auc": float(np.mean(scores["test_roc_auc"])),
        "seconds": float(np.sum(scores["fit_time"]) + np.sum(scores["score_time"])),
    }


def plain(value: object) -> object:
    """A numpy scalar as the Python number it holds; anything else unchanged."""
    # scipy draws numpy scalars, whose repr would clutter every logged candidate.
    return value.item() if isinstance(value, np.generic) else value


def sklearn_search(method: Method, X: pd.DataFrame, y: pd.Series) -> list[dict]:
    """
    Run GridSearchCV or RandomizedSearchCV and return its candidates.

    Args:
        method: A "grid" or "random" method.
        X: Features to search on.
        y: Labels.

    Returns:
        One history entry per candidate, in the order the searcher proposed them.
    """
    common = {
        "scoring": SCORING,
        "refit": False,
        "cv": inner_folds(),
        "n_jobs": N_JOBS,
    }
    if method.kind == "grid":
        grid = {f"model__{name}": values for name, values in GRID.items()}
        searcher = GridSearchCV(build_tuned({}), grid, **common)
    else:
        searcher = RandomizedSearchCV(
            build_tuned({}),
            distributions(),
            n_iter=method.n_iter,
            random_state=SEED,
            **common,
        )
    searcher.fit(X, y)

    res = searcher.cv_results_
    folds = INNER_FOLDS
    history = []
    for i, params in enumerate(res["params"]):
        history.append(
            {
                "params": {
                    k.removeprefix("model__"): plain(v) for k, v in params.items()
                },
                "pr_auc": float(res["mean_test_pr_auc"][i]),
                # cv_results_ reports the population deviation; the sample one
                # matches what cross_validate() entries carry.
                "pr_auc_std": float(
                    res["std_test_pr_auc"][i] * np.sqrt(folds / (folds - 1))
                ),
                "roc_auc": float(res["mean_test_roc_auc"][i]),
                "seconds": float(
                    (res["mean_fit_time"][i] + res["mean_score_time"][i]) * folds
                ),
            }
        )
    return history


def bayes_search(method: Method, X: pd.DataFrame, y: pd.Series) -> list[dict]:
    """
    Run Optuna's TPE over the GRID intervals and return its trials.

    A trial's next proposal depends on every result before it, so trials run
    one after another; the parallelism is inside a trial, one fold per core.

    Args:
        method: A "bayes" method.
        X: Features to search on.
        y: Labels.

    Returns:
        One history entry per trial, in the order Optuna proposed them.
    """
    # Optuna's own handler would print every trial to stderr and bypass the run
    # log; its records go through the root logger instead, and only warnings,
    # since each trial is logged below with the scores the report uses.
    disable_default_handler()
    enable_propagation()
    set_verbosity(OPTUNA_WARNING)

    folds = inner_folds()
    history: list[dict] = []

    def objective(trial: Trial) -> float:
        params = {name: suggest(trial, name) for name in GRID}
        scores = cross_validate(
            build_tuned(params), X, y, scoring=SCORING, cv=folds, n_jobs=N_JOBS
        )
        history.append(candidate(params, scores))
        log.debug("Trial %d: %s", trial.number, history[-1])
        return history[-1][SELECT]

    # multivariate=True is Optuna 5's default for one objective; it is spelled
    # out because the learning rate and the tree size only make sense together.
    sampler = TPESampler(n_startup_trials=STARTUP_TRIALS, multivariate=True, seed=SEED)
    study = create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=method.n_iter)
    return history


def search(method: Method, X: pd.DataFrame, y: pd.Series) -> SearchResult:
    """
    Choose the boosting's hyperparameters with one strategy.

    Args:
        method: The strategy.
        X: Features to search on.
        y: Labels.

    Returns:
        The best candidate by mean PR-AUC over the inner folds, the time the
        search took and every candidate it evaluated.
    """
    started = time.perf_counter()
    if method.kind == "bayes":
        history = bayes_search(method, X, y)
    else:
        history = sklearn_search(method, X, y)
    wall = time.perf_counter() - started

    best = max(history, key=lambda h: h[SELECT])
    return SearchResult(
        best_params=best["params"],
        best_score=best[SELECT],
        best_std=best["pr_auc_std"],
        best_roc_auc=best["roc_auc"],
        wall_seconds=wall,
        compute_seconds=sum(h["seconds"] for h in history),
        history=history,
    )


def scored(folds: StratifiedKFold, X: pd.DataFrame, y: pd.Series) -> dict:
    """Lab 2's untuned boosting under one fold scheme, with its timing."""
    started = time.perf_counter()
    scores = cross_validate(
        build_tuned({}), X, y, scoring=SCORING, cv=folds, n_jobs=N_JOBS
    )
    wall = time.perf_counter() - started
    return {
        **candidate(baseline_params(), scores),
        "folds_pr_auc": scores["test_pr_auc"],
        "wall_seconds": wall,
    }


def run_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """
    Score lab 2's boosting, untuned, under the inner and the outer fold scheme.

    The inner score compares with the best score of a search, the outer one
    with the nested estimate. With no hyperparameter chosen on the folds, the
    outer score is itself unbiased. Writes baseline.json.
    """
    inner = scored(inner_folds(), X_train, y_train)
    outer = scored(outer_folds(), X_train, y_train)
    for name, result in (("inner", inner), ("outer", outer)):
        log.info(
            "Baseline %-5s PR-AUC %.4f ± %.4f  ROC-AUC %.4f  %5.1fs wall",
            name,
            result["pr_auc"],
            result["pr_auc_std"],
            result["roc_auc"],
            result["wall_seconds"],
        )
    save_json(
        {
            "params": baseline_params(),
            "inner_folds": INNER_FOLDS,
            "outer_folds": CV_FOLDS,
            "train_rows": len(X_train),
            "positive_pct": y_train.mean() * 100,
            "inner": inner,
            "outer": outer,
        },
        BASELINE_JSON,
    )


def run_tuning(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Runs every search on the whole training part; writes tuning.json and two figures."""
    results: dict[str, dict] = {}
    for method in METHODS:
        log.info("Searching: %s", method.label)
        result = search(method, X_train, y_train)
        log.info(
            "%-24s %3d candidates  best PR-AUC %.4f ± %.4f  ROC-AUC %.4f  "
            "%6.1fs wall  %7.1fs compute  %s",
            method.label,
            len(result.history),
            result.best_score,
            result.best_std,
            result.best_roc_auc,
            result.wall_seconds,
            result.compute_seconds,
            result.best_params,
        )
        results[method.key] = asdict(result)
    plot_progress(results)
    plot_space(results)
    save_json(
        {
            "grid": GRID,
            "inner_folds": INNER_FOLDS,
            "startup_trials": STARTUP_TRIALS,
            "methods": {m.key: m.label for m in METHODS},
            "results": results,
        },
        TUNING_JSON,
    )


def best_so_far(history: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative compute seconds and the best PR-AUC reached by then."""
    seconds = np.cumsum([h["seconds"] for h in history])
    best = np.maximum.accumulate([h[SELECT] for h in history])
    return seconds, best


def plot_progress(results: dict[str, dict]) -> None:
    """Best PR-AUC found so far against the compute time spent, per method."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for method in METHODS:
        history = results[method.key]["history"]
        seconds, best = best_so_far(history)
        ax.scatter(
            seconds,
            [h[SELECT] for h in history],
            s=10,
            color=method.color,
            alpha=0.35,
        )
        ax.step(
            seconds, best, where="post", color=method.color, lw=2, label=method.label
        )
    ax.set_xscale("log")
    ax.set_xlabel("Сумарний час навчання й оцінювання кандидатів, с (лог. шкала)")
    ax.set_ylabel("PR-AUC на внутрішній CV (3 блоки)")
    ax.set_title("Найкраща знайдена якість залежно від витраченого часу")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9)
    # The weakest candidates sit far below the interesting range and would
    # flatten the curves; their dots fall off the bottom edge.
    top = max(r["best_score"] for r in results.values())
    ax.set_ylim(top - 0.03, top + 0.004)
    save_figure(fig, LAB3 / "tuning_progress.png")


def plot_space(results: dict[str, dict]) -> None:
    """Where each method looked: learning rate against leaf count, coloured by PR-AUC."""
    shown = [METHOD_BY_KEY[k] for k in ("grid", "random-50", "bayes-20")]
    scores = [h[SELECT] for m in shown for h in results[m.key]["history"]]
    # Colour by rank of interest: the bottom of the scale is clipped at the
    # 10th percentile so that a few failed candidates do not wash out the rest.
    norm = Normalize(vmin=float(np.percentile(scores, 10)), vmax=max(scores))
    fig, axes = plt.subplots(1, len(shown), figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, method in zip(axes, shown):
        history = sorted(results[method.key]["history"], key=lambda h: h[SELECT])
        lr = [h["params"]["learning_rate"] for h in history]
        leaves = [h["params"]["max_leaf_nodes"] for h in history]
        points = ax.scatter(
            lr,
            leaves,
            c=[h[SELECT] for h in history],
            cmap="viridis",
            norm=norm,
            s=40,
            edgecolors="black",
            linewidths=0.3,
        )
        best = results[method.key]["best_params"]
        ax.scatter(
            best["learning_rate"],
            best["max_leaf_nodes"],
            marker="*",
            s=320,
            color="red",
            edgecolors="black",
            label="найкращий кандидат",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Log axes label every minor tick in scientific notation, which crowds
        # these short ranges into an unreadable row; plain ticks at round values.
        ax.minorticks_off()
        lr_ticks = [0.02, 0.05, 0.1, 0.2]
        leaf_ticks = [8, 16, 32, 64, 128]
        ax.set_xticks(lr_ticks, [str(t) for t in lr_ticks])
        ax.set_yticks(leaf_ticks, [str(t) for t in leaf_ticks])
        ax.set_title(method.label, fontsize=10)
        ax.set_xlabel("learning_rate")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("max_leaf_nodes")
    # The Bayesian panel's top right corner stays empty: TPE leaves the large,
    # fast-learning trees early. The grid's corner holds one of its points.
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.colorbar(points, ax=axes, label="PR-AUC на внутрішній CV", shrink=0.9)
    save_figure(fig, LAB3 / "tuning_space.png")
