import random
import numpy as np


# ============================================
# 1. Roulette Wheel Selection (RWS)
# ============================================
def rws_selection(population, fitness_values):
    total_fit = sum(fitness_values)
    pick = random.uniform(0, total_fit)
    
    current = 0
    for chromosome, fit in zip(population, fitness_values):
        current += fit
        if current >= pick:
            return chromosome
    return population[-1]


# ============================================
# 2. Stochastic Universal Sampling (SUS)
# ============================================
def sus_selection(population, fitness_values):
    total_fit = sum(fitness_values)
    n = len(population)
    point_distance = total_fit / n
    start_point = random.uniform(0, point_distance)
    points = [start_point + i * point_distance for i in range(n)]

    parents = []
    cumulative = 0
    idx = 0

    for p in points:
        while cumulative < p and idx < len(fitness_values):
            cumulative += fitness_values[idx]
            idx += 1
        parents.append(population[idx - 1])
    
    return parents[0]   # ambil satu parent sesuai desain GA


# ============================================
# 3. Tournament Selection (TS)
# ============================================
def tournament_selection(population, fitness_values, k=3):
    selected = random.sample(list(zip(population, fitness_values)), k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]


# ============================================
# 4. Rank Selection (RS)
# ============================================
def rank_selection(population, fitness_values):
    n = len(population)

    # Ranking: urutkan berdasarkan fitness
    rank_idx = np.argsort(fitness_values)
    ranks = np.empty_like(rank_idx)
    ranks[rank_idx] = np.arange(1, n + 1)

    # Probabilitas berdasarkan ranking
    probs = ranks / ranks.sum()

    return population[np.random.choice(n, p=probs)]
