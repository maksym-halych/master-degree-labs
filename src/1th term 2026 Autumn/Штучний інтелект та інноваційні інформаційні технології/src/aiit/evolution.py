"""A genetic algorithm over a discrete hyperparameter space (lab 3).

A genome is a tuple of indices, one per gene, into that gene's list of allowed
values. Keeping indices rather than the values themselves lets every operator
below treat a learning rate, a layer count and an activation name alike.

Fitness is minimized. The module knows nothing about models: it receives the
fitness as a function that scores a whole batch of genomes, which leaves the
caller free to train that batch in parallel.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np

log = logging.getLogger(__name__)

Genome = tuple[int, ...]
BatchFitness = Callable[[list[Genome]], list[float]]


@dataclass(frozen=True)
class Gene:
    """
    One hyperparameter and the values it may take.

    Attributes:
        name: Name of the hyperparameter.
        values: Allowed values; for an ordered gene, in increasing order.
        ordered: Whether neighbouring values are close in effect, as with a
            learning rate or a layer width. Mutation then steps to a neighbour
            instead of jumping anywhere, so a good value gets refined rather
            than thrown away.
    """

    name: str
    values: tuple
    ordered: bool


@dataclass(frozen=True)
class GAConfig:
    """
    Parameters of the genetic algorithm.

    Attributes:
        population_size: Genomes per generation.
        generations: Generations to run, the random initial one included.
        elite: Best genomes copied unchanged into the next generation.
        tournament_size: Contenders drawn for each tournament.
        crossover_rate: Probability that a child is bred by crossover rather
            than copied from its first parent.
        mutation_rate: Probability that each gene of a child mutates.
        min_diversity: Diversity below which the offspring count as
            degenerated and random immigrants are brought in; 0 never does.
        immigrants: Offspring replaced by random genomes when that happens.
        reject_duplicates: Whether a child that copies a genome already in
            the next generation is mutated until it no longer does.
    """

    population_size: int
    generations: int
    elite: int
    tournament_size: int
    crossover_rate: float
    mutation_rate: float
    min_diversity: float
    immigrants: int
    reject_duplicates: bool

    def __post_init__(self) -> None:
        if self.elite + self.immigrants > self.population_size:
            raise ValueError("Immigrants would replace part of the elite")
        if self.tournament_size > self.population_size:
            raise ValueError("A tournament cannot be larger than the population")


@dataclass(frozen=True)
class Generation:
    """
    One generation of a run, after it was scored.

    Attributes:
        index: Generation number, 0 for the random initial population.
        population: The genomes of the generation.
        fitness: Fitness of each genome, in the same order.
        diversity: Diversity of the population as it was scored.
        offspring_diversity: Diversity of the offspring before immigrants were
            brought in; equals `diversity` when none were.
        immigrants: Random genomes brought in to restore diversity.
        evaluations: Unique genomes scored by the run so far — its real cost.
    """

    index: int
    population: list[Genome]
    fitness: list[float]
    diversity: float
    offspring_diversity: float
    immigrants: int
    evaluations: int


@dataclass(frozen=True)
class SearchResult:
    """
    Everything a search scored.

    Attributes:
        evaluations: Every unique genome scored, in the order it was scored.
        generations: Per-generation snapshots; empty for random search.
    """

    evaluations: list[tuple[Genome, float]]
    generations: list[Generation]

    @property
    def best(self) -> tuple[Genome, float]:
        """The fittest genome found and its fitness."""
        return min(self.evaluations, key=lambda e: e[1])


class _Scorer:
    """
    Scores genomes, each unique one only once.

    The caller's training is seeded, so a genome always gets the same fitness:
    reusing it is exact, and the elite that survives into the next generation
    costs no training at all.
    """

    def __init__(self, fitness: BatchFitness) -> None:
        self._fitness = fitness
        self._cache: dict[Genome, float] = {}
        self.evaluations: list[tuple[Genome, float]] = []

    def __call__(self, genomes: list[Genome]) -> list[float]:
        new = [g for g in dict.fromkeys(genomes) if g not in self._cache]
        if new:
            for genome, score in zip(new, self._fitness(new), strict=True):
                self._cache[genome] = score
                self.evaluations.append((genome, score))
        return [self._cache[g] for g in genomes]


def decode(space: tuple[Gene, ...], genome: Genome) -> dict:
    """Map a genome to the hyperparameter values it encodes, by gene name."""
    return {gene.name: gene.values[i] for gene, i in zip(space, genome, strict=True)}


def random_genome(space: tuple[Gene, ...], rng: np.random.Generator) -> Genome:
    """Draw every gene uniformly from its allowed values."""
    return tuple(int(rng.integers(len(gene.values))) for gene in space)


def _unique_random(
    space: tuple[Gene, ...], count: int, rng: np.random.Generator
) -> list[Genome]:
    genomes: list[Genome] = []
    while len(genomes) < count:
        genome = random_genome(space, rng)
        if genome not in genomes:
            genomes.append(genome)
    return genomes


def diversity(population: list[Genome]) -> float:
    """
    Mean share of genes in which two members of the population differ.

    Args:
        population: The genomes to compare, all of the same length.

    Returns:
        A value in [0, 1]. Zero means every member is the same genome: the
        population has degenerated, and crossover can no longer produce
        anything its parents did not already have.
    """
    pairs = list(combinations(population, 2))
    if not pairs:
        return 0.0
    n_genes = len(population[0])
    return float(
        np.mean(
            [sum(x != y for x, y in zip(a, b, strict=True)) / n_genes for a, b in pairs]
        )
    )


def tournament(
    population: list[Genome],
    fitness: list[float],
    size: int,
    rng: np.random.Generator,
) -> Genome:
    """
    Tournament selection: the fittest of `size` members drawn at random.

    Only the ranking of the contenders matters, not how far apart their RMSEs
    are, so a single outstanding genome cannot take over the population the way
    it would under roulette selection.

    Args:
        population: Genomes to select from.
        fitness: Fitness of each genome; lower is better.
        size: Contenders per tournament; larger means stronger selection.
        rng: Random number generator.

    Returns:
        The winning genome.
    """
    contenders = rng.choice(len(population), size=size, replace=False)
    winner = min(contenders, key=lambda i: fitness[i])
    return population[winner]


def uniform_crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    """
    Take every gene of the child from either parent with equal probability.

    The order of hyperparameters along the genome means nothing, so one-point
    crossover, which keeps neighbouring genes together, would favour pairs of
    hyperparameters for no reason.
    """
    from_a = rng.random(len(a)) < 0.5
    return tuple(x if keep else y for x, y, keep in zip(a, b, from_a, strict=True))


def _mutate_gene(gene: Gene, index: int, rng: np.random.Generator) -> int:
    n = len(gene.values)
    if gene.ordered:
        step = int(rng.choice((-1, 1)))
        # At either end of the range the only neighbour is inward.
        if not 0 <= index + step < n:
            step = -step
        return index + step
    # Any value but the current one, uniformly.
    other = int(rng.integers(n - 1))
    return other if other < index else other + 1


def mutate(
    genome: Genome,
    space: tuple[Gene, ...],
    rate: float,
    rng: np.random.Generator,
) -> Genome:
    """
    Mutate every gene independently with probability `rate`.

    An ordered gene steps to a neighbouring value; any other gene jumps to a
    different value drawn uniformly.
    """
    return tuple(
        _mutate_gene(gene, i, rng) if rng.random() < rate else i
        for gene, i in zip(space, genome, strict=True)
    )


def _mutate_one(
    genome: Genome, space: tuple[Gene, ...], rng: np.random.Generator
) -> Genome:
    position = int(rng.integers(len(genome)))
    genes = list(genome)
    genes[position] = _mutate_gene(space[position], genes[position], rng)
    return tuple(genes)


def _next_generation(
    population: list[Genome],
    fitness: list[float],
    space: tuple[Gene, ...],
    config: GAConfig,
    rng: np.random.Generator,
) -> tuple[list[Genome], float, int]:
    ranked = sorted(range(len(population)), key=lambda i: fitness[i])
    # Elitism: the best genomes survive unchanged, so the best fitness of the
    # population can never get worse from one generation to the next.
    children = [population[i] for i in ranked[: config.elite]]

    while len(children) < config.population_size:
        a = tournament(population, fitness, config.tournament_size, rng)
        b = tournament(population, fitness, config.tournament_size, rng)
        child = (
            uniform_crossover(a, b, rng) if rng.random() < config.crossover_rate else a
        )
        child = mutate(child, space, config.mutation_rate, rng)
        # A copy of a genome already in the generation teaches the search
        # nothing and is the first step towards degeneration.
        while config.reject_duplicates and child in children:
            child = _mutate_one(child, space, rng)
        children.append(child)

    offspring_diversity = diversity(children)
    if offspring_diversity >= config.min_diversity:
        return children, offspring_diversity, 0

    # The population has nearly collapsed onto one genome. Random immigrants
    # take the last places — never the elite's — and bring back the variety
    # crossover needs.
    for slot in range(
        config.population_size - config.immigrants, config.population_size
    ):
        immigrant = random_genome(space, rng)
        while immigrant in children:
            immigrant = random_genome(space, rng)
        children[slot] = immigrant
    return children, offspring_diversity, config.immigrants


def evolve(
    space: tuple[Gene, ...],
    fitness: BatchFitness,
    config: GAConfig,
    rng: np.random.Generator,
) -> SearchResult:
    """
    Run the genetic algorithm.

    Each generation is scored, its best genomes are kept as the elite, and the
    rest of the next generation is bred by tournament selection, uniform
    crossover and mutation. Whenever the offspring lose too much diversity,
    random immigrants replace part of them.

    Args:
        space: The genes a genome is made of.
        fitness: Scores a batch of genomes; lower is better.
        config: Parameters of the algorithm.
        rng: Random number generator; the run is reproducible given its seed.

    Returns:
        Every genome scored and a snapshot of every generation.
    """
    score = _Scorer(fitness)
    population = _unique_random(space, config.population_size, rng)
    offspring_diversity = diversity(population)
    immigrants = 0
    generations: list[Generation] = []

    for index in range(config.generations):
        scores = score(population)
        generation = Generation(
            index=index,
            population=population,
            fitness=scores,
            diversity=diversity(population),
            offspring_diversity=offspring_diversity,
            immigrants=immigrants,
            evaluations=len(score.evaluations),
        )
        generations.append(generation)
        log.info(
            "Generation %d: best=%.4f median=%.4f diversity=%.2f immigrants=%d evaluations=%d",
            index,
            min(scores),
            float(np.median(scores)),
            generation.diversity,
            immigrants,
            generation.evaluations,
        )

        if index < config.generations - 1:
            population, offspring_diversity, immigrants = _next_generation(
                population, scores, space, config, rng
            )

    return SearchResult(score.evaluations, generations)


def random_search(
    space: tuple[Gene, ...],
    fitness: BatchFitness,
    budget: int,
    rng: np.random.Generator,
) -> SearchResult:
    """
    Score `budget` distinct genomes drawn uniformly — the baseline a GA must beat.

    Args:
        space: The genes a genome is made of.
        fitness: Scores a batch of genomes; lower is better.
        budget: Number of distinct genomes to score.
        rng: Random number generator.

    Returns:
        Every genome scored, in the order it was drawn.
    """
    score = _Scorer(fitness)
    score(_unique_random(space, budget, rng))
    return SearchResult(score.evaluations, [])
