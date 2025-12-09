import numpy as np
import random


# ============================================
# 1. Menghitung nilai M (Mean), L (Loss), K (Kurtosis-like)
# ============================================
def calculate_statistics(fitness_values):
    fitness = np.array(fitness_values)

    M = np.mean(fitness)
    L = np.max(fitness) - np.min(fitness)       # loss range fitness
    K = np.std(fitness)                         # standard deviation

    return M, L, K


# ============================================
# 2. Dynamic Decision Matrix (DDM)
#    Baris = alternatif (4 metode seleksi)
#    Kolom = criteria M, L, K
# ============================================
def build_ddm(M, L, K):
    # asumsikan semua metode seleksi punya nilai sama (di paper begitu)
    # Pembeda hanya pada TOPSIS weighting & rank
    ddm = np.array([
        [M, L, K],   # RWS
        [M, L, K],   # SUS
        [M, L, K],   # Tournament
        [M, L, K],   # Rank Selection
    ], dtype=float)

    return ddm


# ============================================
# 3. Weight / Bobot kriteria (mengikuti paper)
#    Jika paper menyebut M,L,K itu benefit, kita beri bobot rata
# ============================================
def get_weights():
    # M, L, K diberi bobot sama
    return np.array([1/3, 1/3, 1/3])


# ============================================
# 4. TOPSIS utama
# ============================================
def topsis_selection(ddm, weights):
    # Normalisasi
    norm = ddm / np.sqrt((ddm**2).sum(axis=0))

    # Pembobotan
    weighted = norm * weights

    # Ideal positif & negatif
    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)

    # Jarak ke ideal
    dist_best = np.sqrt(((weighted - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst)**2).sum(axis=1))

    # Skor TOPSIS
    scores = dist_worst / (dist_best + dist_worst + 1e-9)

    # Index metode terbaik
    best_idx = np.argmax(scores)

    return best_idx, scores


# ============================================
# 5. Wrapper untuk digunakan GA
# ============================================
def select_best_method(fitness_values):
    M, L, K = calculate_statistics(fitness_values)
    
    # Jika semua nol / identik → fallback random (menghindari pembagian nol)
    if (M == 0 and L == 0 and K == 0) or np.isnan(M) or np.isnan(L) or np.isnan(K):
        methods = ["RWS", "SUS", "TS", "RS"]
        return random.choice(methods), np.array([0, 0, 0, 0])

    ddm = build_ddm(M, L, K)
    weights = get_weights()
    
    best_idx, scores = topsis_selection(ddm, weights)

    methods = ["RWS", "SUS", "TS", "RS"]

    return methods[best_idx], scores
