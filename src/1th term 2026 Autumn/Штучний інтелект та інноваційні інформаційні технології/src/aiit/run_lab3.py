"""Lab 3: evolutionary hyperparameter search for the lab 2 network (variant 1).

A genetic algorithm (aiit.evolution) tunes the PyTorch feedforward network of
lab 2 on California Housing, with the validation RMSE as the fitness. Random
search over the same number of trainings is the baseline the GA has to beat,
and baseline-relu, the lab 2 winner, is the reference point for both. A naive
GA, stripped of its defences against degeneration, shows what they prevent.

Every search runs once from SEED for the detailed figures, then again from ten
search seeds, so that the comparison of methods rests on more than one run.

The test sample is used once, after both searches are over, for the three final
configurations: selecting on it would make the test RMSE one more fitted value.

The figure titles and gene labels passed to aiit.plots stay in Ukrainian: they
are rendered into the figures the report shows.
"""

import json
import logging
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np
import torch
import torch.nn as nn

from aiit.config import EPOCHS, LAB3, SEED
from aiit.data import prepare
from aiit.evolution import (
    BatchFitness,
    GAConfig,
    Gene,
    Genome,
    SearchResult,
    decode,
    evolve,
    random_search,
)
from aiit.metrics import regression_metrics
from aiit.plots import (
    ga_convergence,
    gene_evolution,
    predicted_vs_actual,
    search_comparison,
    seed_study,
)
from aiit.torch_models import build_mlp

log = logging.getLogger(__name__)

SEARCH_SPACE: tuple[Gene, ...] = (
    Gene("n_layers", (1, 2, 3), ordered=True),
    # Width of the first hidden layer; every next one is half as wide.
    Gene("width", (32, 64, 128, 256), ordered=True),
    Gene("activation", ("relu", "gelu", "leakyrelu"), ordered=False),
    Gene("dropout", (0.0, 0.1, 0.2, 0.3), ordered=True),
    Gene("batch_norm", (False, True), ordered=False),
    Gene("weight_decay", (0.0, 1e-5, 1e-4, 1e-3), ordered=True),
    Gene("learning_rate", (3e-4, 1e-3, 3e-3, 1e-2), ordered=True),
    Gene("batch_size", (64, 128, 256, 512), ordered=True),
)

GENE_LABELS = {
    "n_layers": "Кількість шарів",
    "width": "Ширина 1-го шару",
    "activation": "Активація",
    "dropout": "Dropout",
    "batch_norm": "BatchNorm",
    "weight_decay": "Weight decay (L2)",
    "learning_rate": "Швидкість навчання",
    "batch_size": "Розмір батча",
}

METHOD_LABELS = {
    "genetic": "Генетичний алгоритм",
    "naive-genetic": "ГА без захисту від виродження",
    "random-search": "Випадковий пошук",
}

# baseline-relu of lab 2 as a genome: hidden=(64, 32), ReLU, no Dropout, no
# BatchNorm, no weight decay, Adam at 1e-3, batches of 256.
LAB2_BASELINE: Genome = (1, 1, 0, 0, 0, 0, 1, 2)

GA = GAConfig(
    population_size=12,
    generations=15,
    elite=2,
    tournament_size=3,
    crossover_rate=0.9,
    mutation_rate=1 / len(SEARCH_SPACE),
    min_diversity=0.3,
    immigrants=3,
    reject_duplicates=True,
)

# The same algorithm with both of its defences against degeneration switched
# off; it exists to show what they defend against.
NAIVE_GA = replace(GA, min_diversity=0.0, immigrants=0, reject_duplicates=False)

# Seeds of the search itself, not of the training, which stays seeded with SEED:
# every genome keeps its fitness, only the path the search takes changes.
STUDY_SEEDS = tuple(range(SEED, SEED + 10))

# Leaves headroom for the rest of the machine; each worker holds its own copy
# of PyTorch and of the data.
WORKERS = min(6, os.cpu_count() or 1)

# Filled once per worker process by _init_worker: spawned workers inherit
# nothing from the parent.
_DATA: dict = {}


def _init_worker() -> None:
    # One thread per worker: networks this small gain nothing from intra-op
    # parallelism, and a fixed thread count keeps every training bit-for-bit
    # reproducible whichever worker it lands on.
    torch.set_num_threads(1)
    _DATA.update(prepare())


def hidden_layers(params: dict) -> tuple[int, ...]:
    """Widths of the hidden layers: the first one given, each next one halved."""
    return tuple(params["width"] // 2**i for i in range(params["n_layers"]))


def train(params: dict, ds: dict) -> nn.Module:
    """
    Train the network a set of hyperparameters describes.

    Seeded, so the same set always yields the same model — which is what lets
    the GA score every genome only once.

    Args:
        params: Decoded genome, keyed by the gene names of SEARCH_SPACE.
        ds: The prepared dataset, as returned by aiit.data.prepare.

    Returns:
        The trained network, switched to eval mode.

    Raises:
        ValueError: If the activation function is unknown.
    """
    torch.manual_seed(SEED)
    model = build_mlp(
        ds["X_train"].shape[1],
        hidden_layers(params),
        params["activation"],
        params["dropout"],
        params["batch_norm"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    criterion = nn.MSELoss()

    X = torch.from_numpy(ds["X_train"])
    y = torch.from_numpy(ds["y_train"]).unsqueeze(1)
    generator = torch.Generator().manual_seed(SEED)
    batch_size = params["batch_size"]

    model.train()
    for _ in range(EPOCHS):
        order = torch.randperm(len(X), generator=generator)
        for start in range(0, len(X), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            criterion(model(X[batch]), y[batch]).backward()
            optimizer.step()

    model.eval()
    return model


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Predictions as a flat array; the model must already be in eval mode."""
    return model(torch.from_numpy(X)).numpy().ravel()


def validation_rmse(genome: Genome) -> float:
    """
    Fitness of a genome: the validation RMSE of the network it encodes.

    Runs in a worker process. A training that diverged scores +inf, so
    selection discards it instead of comparing NaNs.
    """
    model = train(decode(SEARCH_SPACE, genome), _DATA)
    pred = predict(model, _DATA["X_val"])
    if not np.all(np.isfinite(pred)):
        return math.inf
    return float(np.sqrt(np.mean((pred - _DATA["y_val"]) ** 2)))


def final_evaluation(genome: Genome) -> dict:
    """
    Retrain a genome and measure it on the validation and test samples.

    Runs in a worker process, under the same single thread as the search, so
    the validation RMSE matches the fitness the search saw.
    """
    params = decode(SEARCH_SPACE, genome)
    start = time.perf_counter()
    model = train(params, _DATA)
    elapsed = time.perf_counter() - start
    test_pred = predict(model, _DATA["X_test"])

    return {
        "params": params,
        "hidden": hidden_layers(params),
        "val": regression_metrics(_DATA["y_val"], predict(model, _DATA["X_val"])),
        "test": regression_metrics(_DATA["y_test"], test_pred),
        "train_time_s": round(elapsed, 2),
        "n_params": sum(p.numel() for p in model.parameters()),
        "test_pred": test_pred,
    }


def _search_summary(result: SearchResult, wall_time: float) -> dict:
    genome, fitness = result.best
    return {
        "evaluations": len(result.evaluations),
        "wall_time_s": round(wall_time, 1),
        "best_val_rmse": fitness,
        "best_params": decode(SEARCH_SPACE, genome),
        "fitness": [f for _, f in result.evaluations],
    }


def _generation_summary(result: SearchResult) -> list[dict]:
    return [
        {
            "index": g.index,
            "best": min(g.fitness),
            "median": float(np.median(g.fitness)),
            "worst": max(g.fitness),
            "diversity": g.diversity,
            "offspring_diversity": g.offspring_diversity,
            "immigrants": g.immigrants,
            "evaluations": g.evaluations,
            "best_params": decode(
                SEARCH_SPACE, g.population[int(np.argmin(g.fitness))]
            ),
            "population": [list(genome) for genome in g.population],
        }
        for g in result.generations
    ]


def _open_pool() -> ProcessPoolExecutor:
    # spawn rather than fork: forking a process that already runs PyTorch's
    # thread pools can deadlock the child.
    return ProcessPoolExecutor(
        WORKERS,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
    )


def _batch_fitness(pool: ProcessPoolExecutor) -> BatchFitness:
    def fitness(genomes: list[Genome]) -> list[float]:
        scores = list(pool.map(validation_rmse, genomes))
        for genome, score in zip(genomes, scores, strict=True):
            log.debug("RMSE(val)=%.4f  %s", score, decode(SEARCH_SPACE, genome))
        return scores

    return fitness


def _run_searches(
    fitness: BatchFitness, seed: int
) -> dict[str, tuple[SearchResult, float]]:
    """Run the GA, the naive GA and random search from one search seed.

    Returns each search's result and wall time in seconds, keyed by method.
    """
    searches: dict[str, tuple[SearchResult, float]] = {}
    for name, config in (("genetic", GA), ("naive-genetic", NAIVE_GA)):
        start = time.perf_counter()
        result = evolve(SEARCH_SPACE, fitness, config, np.random.default_rng(seed))
        searches[name] = (result, time.perf_counter() - start)

    # Random search gets exactly as many trainings as the GA spent. The same
    # seed makes it draw the GA's initial population first, so the two differ
    # only in what happens after that common start.
    start = time.perf_counter()
    budget = len(searches["genetic"][0].evaluations)
    result = random_search(SEARCH_SPACE, fitness, budget, np.random.default_rng(seed))
    searches["random-search"] = (result, time.perf_counter() - start)

    for name, (result, elapsed) in searches.items():
        log.info(
            "Seed %d, %s: best RMSE(val)=%.4f after %d trainings, %.0f s",
            seed,
            name,
            result.best[1],
            len(result.evaluations),
            elapsed,
        )
    return searches


def run_evolutionary_search() -> None:
    """Runs every search once, retrains the finalists, writes figures and metrics."""
    log.info(
        "Search space: %d configurations",
        math.prod(len(g.values) for g in SEARCH_SPACE),
    )
    log.info("GA: %s", GA)
    log.info("Naive GA: %s", NAIVE_GA)
    log.info("Workers: %d", WORKERS)

    with _open_pool() as pool:
        searches = _run_searches(_batch_fitness(pool), SEED)
        finalists = {"lab2-baseline": LAB2_BASELINE} | {
            name: result.best[0] for name, (result, _) in searches.items()
        }
        finals = dict(
            zip(finalists, pool.map(final_evaluation, finalists.values()), strict=True)
        )

    for name, r in finals.items():
        log.info(
            "%s: RMSE(val)=%.4f RMSE(test)=%.4f R2(test)=%.4f params=%d %s",
            name,
            r["val"]["rmse"],
            r["test"]["rmse"],
            r["test"]["r2"],
            r["n_params"],
            r["params"],
        )

    ga, naive, rs = (
        searches[n][0] for n in ("genetic", "naive-genetic", "random-search")
    )
    generations = _generation_summary(ga)
    reference = finals["lab2-baseline"]["val"]["rmse"]
    ga_convergence(
        generations,
        _generation_summary(naive),
        reference,
        GA.min_diversity,
        LAB3 / "ga_convergence.png",
    )
    search_comparison(
        {
            METHOD_LABELS[name]: [f for _, f in result.evaluations]
            for name, (result, _) in searches.items()
        },
        reference,
        LAB3 / "search_comparison.png",
    )
    gene_evolution(
        [g["population"] for g in generations],
        [(GENE_LABELS[gene.name], gene.values) for gene in SEARCH_SPACE],
        LAB3 / "gene_evolution.png",
    )
    predicted_vs_actual(
        prepare()["y_test"],
        finals["genetic"]["test_pred"],
        LAB3 / "pred_vs_actual.png",
        "Найкраща конфігурація генетичного алгоритму",
    )

    for r in finals.values():
        del r["test_pred"]
    LAB3.mkdir(parents=True, exist_ok=True)
    (LAB3 / "lab3.json").write_text(
        json.dumps(
            {
                "search_space": {g.name: list(g.values) for g in SEARCH_SPACE},
                "ga_config": asdict(GA),
                "naive_ga_config": asdict(NAIVE_GA),
                "workers": WORKERS,
                "generations": generations,
                "naive_generations": _generation_summary(naive),
                "searches": {
                    name: _search_summary(result, elapsed)
                    for name, (result, elapsed) in searches.items()
                },
                "final": finals,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_seed_study() -> None:
    """Repeats every search from several seeds and measures each winner on test.

    One run cannot tell a better search method from a luckier one; ten runs
    from different seeds can.
    """
    runs: list[dict] = []
    with _open_pool() as pool:
        baseline = pool.submit(final_evaluation, LAB2_BASELINE).result()
        fitness = _batch_fitness(pool)
        for seed in STUDY_SEEDS:
            searches = _run_searches(fitness, seed)
            winners = {name: result.best[0] for name, (result, _) in searches.items()}
            tested = dict(
                zip(winners, pool.map(final_evaluation, winners.values()), strict=True)
            )
            run: dict = {"seed": seed}
            for name, (result, elapsed) in searches.items():
                generations = result.generations
                run[name] = {
                    "best_val_rmse": result.best[1],
                    "test_rmse": tested[name]["test"]["rmse"],
                    "best_params": tested[name]["params"],
                    "evaluations": len(result.evaluations),
                    "wall_time_s": round(elapsed, 1),
                    "min_diversity": min(
                        (g.diversity for g in generations), default=None
                    ),
                    "final_diversity": generations[-1].diversity
                    if generations
                    else None,
                    "immigrant_events": sum(1 for g in generations if g.immigrants),
                    "fitness": [f for _, f in result.evaluations],
                }
            runs.append(run)

    summary: dict[str, dict] = {}
    for name in METHOD_LABELS:
        val = np.array([r[name]["best_val_rmse"] for r in runs])
        test = np.array([r[name]["test_rmse"] for r in runs])
        summary[name] = {
            "val_mean": float(val.mean()),
            "val_std": float(val.std(ddof=1)),
            "test_mean": float(test.mean()),
            "test_std": float(test.std(ddof=1)),
            "evaluations_mean": float(np.mean([r[name]["evaluations"] for r in runs])),
        }
        log.info(
            "%s over %d seeds: RMSE(val)=%.4f±%.4f RMSE(test)=%.4f±%.4f",
            name,
            len(runs),
            val.mean(),
            val.std(ddof=1),
            test.mean(),
            test.std(ddof=1),
        )

    # Paired by seed: both searches started from the same initial population.
    wins = {
        sample: sum(r["genetic"][key] < r["random-search"][key] for r in runs)
        for sample, key in (("val", "best_val_rmse"), ("test", "test_rmse"))
    }
    log.info("GA beats random search on val in %d of %d seeds", wins["val"], len(runs))
    log.info(
        "GA beats random search on test in %d of %d seeds", wins["test"], len(runs)
    )

    seed_study(
        runs,
        METHOD_LABELS,
        baseline["val"]["rmse"],
        LAB3 / "seed_study.png",
    )

    (LAB3 / "lab3_seeds.json").write_text(
        json.dumps(
            {
                "seeds": list(STUDY_SEEDS),
                "summary": summary,
                "ga_wins": wins,
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
