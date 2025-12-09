# ga_core_cvrp.py
import random
import numpy as np

# ==============
# Utilities for CVRP
# ==============

def total_distance_cvrp(route, distance_matrix, depot=0):
    """
    Hitung total jarak seluruh rute (route adalah list berisi 0 sebagai separator depot).
    Jika route tidak diakhiri 0, fungsi akan menambahkan return-to-depot otomatis.
    """
    dist = 0.0
    last = depot
    for node in route:
        # node may be 0 (depot marker) or customer index
        dist += distance_matrix[last][node]
        last = node
    # ensure return to depot
    if last != depot:
        dist += distance_matrix[last][depot]
    return dist

def cvrp_fitness(route, distance_matrix, demand, capacity, depot=0, penalty_coef=1000.0):
    """
    route: list with depot markers (0)
    demand: list where index corresponds to node (0 is depot demand 0)
    capacity: scalar capacity per vehicle
    Returns fitness (higher is better): 1 / (total_distance + penalty * overload)
    """
    total_distance = 0.0
    overload = 0.0

    load = 0
    last = depot

    for node in route:
        if node == depot:
            # return to depot from last
            total_distance += distance_matrix[last][depot]
            last = depot
            load = 0
            continue

        # travel to node
        total_distance += distance_matrix[last][node]
        load += demand[node]
        last = node

        if load > capacity:
            overload += (load - capacity)

    # final return to depot
    if last != depot:
        total_distance += distance_matrix[last][depot]

    score = 1.0 / (total_distance + penalty_coef * overload + 1e-9)
    return score

# ==============
# Population init for CVRP
# ==============
def init_cvrp_population(pop_size, num_customers, num_vehicles):
    """
    Customers are 1..num_customers. Depot = 0.
    Chromosome representation: list containing customers and (num_vehicles-1) zeros as separators.
    Example for 6 customers, 2 vehicles:
      [3,1,0,5,2,4,6]  (0 separates route1 and route2)
    """
    population = []
    customers = list(range(1, num_customers + 1))
    for _ in range(pop_size):
        perm = customers.copy()
        random.shuffle(perm)

        # insert num_vehicles-1 depot markers at random positions
        for _ in range(num_vehicles - 1):
            pos = random.randint(0, len(perm))
            perm.insert(pos, 0)

        population.append(perm)
    return population

# ==============
# Repair function to ensure valid chromosome after crossover/mutation
# ==============
def repair_route(route, num_customers, num_vehicles):
    """
    Ensures:
    - Every customer 1..num_customers appears exactly once
    - Exactly num_vehicles-1 depot separators (0)
    - No other numbers out of range
    Approach:
    - Remove any stray numbers not in 0..num_customers
    - Collect customers in order (left to right) without duplicates
    - Insert separators evenly/randomly
    """
    # keep only valid tokens (0..num_customers)
    cleaned = [x for x in route if (x == 0 or (1 <= x <= num_customers))]

    # extract customers preserving first occurrence order
    seen = set()
    customers = []
    for x in cleaned:
        if x != 0 and x not in seen:
            customers.append(x)
            seen.add(x)

    # if some customers missing, append them
    for c in range(1, num_customers + 1):
        if c not in seen:
            customers.append(c)

    # now insert separators: make num_vehicles segments (may be uneven)
    separators_needed = num_vehicles - 1
    if separators_needed <= 0:
        return customers

    # generate separator positions: choose separators_needed distinct positions between 1..len(customers)-1
    if len(customers) == 0:
        # degenerate
        return [0] * separators_needed

    positions = sorted(random.sample(range(1, len(customers)), separators_needed)) if len(customers) > 1 else [1]*separators_needed

    new_route = []
    last_pos = 0
    for pos in positions:
        new_route.extend(customers[last_pos:pos])
        new_route.append(0)
        last_pos = pos
    new_route.extend(customers[last_pos:])

    return new_route

# ==============
# Adapted OX crossover for CVRP:
# operate on customer sequence (remove zeros), perform OX, then re-insert separators and repair
# ==============
def ox_crossover_cvrp(parent1, parent2, num_customers, num_vehicles):
    # extract customer sequences (remove zeros)
    p1 = [x for x in parent1 if x != 0]
    p2 = [x for x in parent2 if x != 0]
    n = len(p1)
    if n <= 1:
        # trivial
        child = p1.copy()
    else:
        a, b = sorted(random.sample(range(n), 2))
        child = [None] * n
        child[a:b] = p1[a:b]
        pos = b
        for city in p2:
            if city not in child:
                if pos >= n:
                    pos = 0
                child[pos] = city
                pos += 1

    # insert separators at random positions then repair
    for _ in range(num_vehicles - 1):
        pos = random.randint(0, len(child))
        child.insert(pos, 0)

    repaired = repair_route(child, num_customers, num_vehicles)
    return repaired

# ==============
# Mutation: inversion on customers then re-insert separators + repair
# ==============
def inversion_mutation_cvrp(chromosome, num_customers, num_vehicles, mut_rate=1.0):
    # work on customer list
    customers = [x for x in chromosome if x != 0]
    if len(customers) <= 1:
        return chromosome.copy()

    if random.random() > mut_rate:
        # no mutation
        mutated = customers
    else:
        a, b = sorted(random.sample(range(len(customers)), 2))
        mutated = customers[:a] + customers[a:b][::-1] + customers[b:]

    # re-insert separators
    for _ in range(num_vehicles - 1):
        pos = random.randint(0, len(mutated))
        mutated.insert(pos, 0)

    repaired = repair_route(mutated, num_customers, num_vehicles)
    return repaired

# ==============
# Replacement: keep best individuals (parents+offspring)
# ==============
def replace_population_cvrp(population, offspring, distance_matrix, demand, capacity):
    combined = population + offspring
    # compute fitness
    fits = [cvrp_fitness(ind, distance_matrix, demand, capacity) for ind in combined]
    # sort by fitness descending
    idx_sorted = np.argsort(fits)[::-1]
    new_pop = [combined[i] for i in idx_sorted[: len(population)]]
    return new_pop
