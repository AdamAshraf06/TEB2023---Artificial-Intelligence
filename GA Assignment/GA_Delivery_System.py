"""
Genetic Algorithm for Delivery Route Optimisation
-------------------------------------------------
Depot = A. Delivery locations = B, C, D, E, F, G.
Goal: find the shortest round trip A -> (B..G in some order) -> A.
"""

import random

# ----------------------------------------------------------------------
# 1. DATA: symmetric distance matrix (km)
# ----------------------------------------------------------------------
LOCATIONS = ["A", "B", "C", "D", "E", "F", "G"]
DISTANCES = {
    "A": {"A": 0, "B": 4, "C": 8, "D": 6, "E": 7, "F": 9, "G": 5},
    "B": {"A": 4, "B": 0, "C": 5, "D": 7, "E": 3, "F": 8, "G": 6},
    "C": {"A": 8, "B": 5, "C": 0, "D": 4, "E": 6, "F": 3, "G": 7},
    "D": {"A": 6, "B": 7, "C": 4, "D": 0, "E": 5, "F": 6, "G": 8},
    "E": {"A": 7, "B": 3, "C": 6, "D": 5, "E": 0, "F": 4, "G": 6},
    "F": {"A": 9, "B": 8, "C": 3, "D": 6, "E": 4, "F": 0, "G": 5},
    "G": {"A": 5, "B": 6, "C": 7, "D": 8, "E": 6, "F": 5, "G": 0},
}
DEPOT = "A"
DELIVERY_STOPS = ["B", "C", "D", "E", "F", "G"]

# ----------------------------------------------------------------------
# 2. GA PARAMETERS
# ----------------------------------------------------------------------
RANDOM_SEED = 7         # fixed seed -> reproducible experiment
POPULATION_SIZE = 10
NUM_GENERATIONS = 100   # assignment requires at least 50
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.3
ELITE_COUNT = 2         # best routes copied unchanged (elitism)


# ----------------------------------------------------------------------
# 3. CHROMOSOME + FITNESS
# ----------------------------------------------------------------------
# A chromosome is a permutation of B-G, e.g. ['C', 'F', 'E', 'B', 'G', 'D'].
# The depot A is added at the start and end only when the route is evaluated,
# so every chromosome is automatically a valid route.

def full_route(chromosome):
    """Add the depot at the start and the end of the chromosome."""
    return [DEPOT] + chromosome + [DEPOT]


def total_distance(chromosome):
    """Total route length in km, including the return leg to the depot."""
    route = full_route(chromosome)
    return sum(DISTANCES[route[i]][route[i + 1]] for i in range(len(route) - 1))


# Shorter distance = better route, so the GA MINIMISES total_distance.


# ----------------------------------------------------------------------
# 4. INITIAL POPULATION
# ----------------------------------------------------------------------
def create_random_route():
    """Random valid route: a shuffled list of B-G (each appears exactly once)."""
    route = DELIVERY_STOPS[:]
    random.shuffle(route)
    return route


def create_initial_population(size):
    return [create_random_route() for _ in range(size)]


# ----------------------------------------------------------------------
# 5. GA OPERATORS
# ----------------------------------------------------------------------
def tournament_selection(population):
    """Parent selection: pick a few random routes, return the shortest one."""
    contestants = random.sample(population, TOURNAMENT_SIZE)
    return min(contestants, key=total_distance)


def order_crossover(parent1, parent2):
    """
    Order Crossover (OX) - keeps the child a valid permutation.
    1. Copy a random slice from parent1 into the child.
    2. Fill the remaining positions with the missing stops,
       in the order they appear in parent2.
    """
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child = [None] * size
    child[start:end + 1] = parent1[start:end + 1]

    remaining = [stop for stop in parent2 if stop not in child]
    fill_positions = [i for i in range(size) if child[i] is None]
    for pos, stop in zip(fill_positions, remaining):
        child[pos] = stop
    return child


def swap_mutation(route):
    """Mutation: with some probability, swap two stops (still a valid route)."""
    route = route[:]  # work on a copy
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route


# ----------------------------------------------------------------------
# 6. MAIN GA LOOP
# ----------------------------------------------------------------------
def run_genetic_algorithm():
    random.seed(RANDOM_SEED)
    population = create_initial_population(POPULATION_SIZE)

    best_route = min(population, key=total_distance)
    print(f"Generation   0 | best distance = {total_distance(best_route)} km")

    for generation in range(1, NUM_GENERATIONS + 1):
        # Sort so the shortest routes come first
        population.sort(key=total_distance)

        # Elitism: copy the best routes unchanged into the next generation
        next_population = [route[:] for route in population[:ELITE_COUNT]]

        # Fill the rest of the new generation with children
        while len(next_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = order_crossover(parent1, parent2)
            child = swap_mutation(child)
            next_population.append(child)

        population = next_population

        current_best = min(population, key=total_distance)
        if total_distance(current_best) < total_distance(best_route):
            best_route = current_best

        if generation % 10 == 0 or generation == 1:
            print(f"Generation {generation:3d} | best distance = "
                  f"{total_distance(best_route)} km")

    return best_route


if __name__ == "__main__":
    best = run_genetic_algorithm()
    print("\n===== RESULT =====")
    print("Best route    :", " -> ".join(full_route(best)))
    print("Total distance:", total_distance(best), "km")