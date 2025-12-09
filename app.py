# app_cvrp.py
import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import time
import base64
from io import BytesIO

# import fungsi CVRP core (harus ada file ga_core_cvrp.py di folder yang sama)
from ga_core import (
    init_cvrp_population,
    cvrp_fitness,
    ox_crossover_cvrp,
    inversion_mutation_cvrp,
    replace_population_cvrp,
    repair_route,
)
from selection_methods import (
    rws_selection,
    sus_selection,
    tournament_selection,
    rank_selection,
)
from topsis_decider import select_best_method

# ---------------------------
# CONFIGURASI TAMPILAN
# ---------------------------
st.set_page_config(page_title="GA + SODGA (CVRP)", layout="wide", page_icon="🧭")

# ===== Styling CSS (lebih modern) =====
st.markdown("""
<style>
.main-title { font-size:28px; font-weight:700; color:#2E7D32; }
.sub { color:#555; }
.card { padding:12px; border-radius:8px; background:#F5F5F5; border-left:4px solid #2E7D32; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("<div class='main-title'>🧭 Genetic Algorithm + SODGA — CVRP</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Vehicle Routing Problem dengan kapasitas, depot, dan permintaan tiap node. (Informasi log & status real-time disediakan.)</div><br>", unsafe_allow_html=True)

# =========================================================
# PEMBUKA (KONTEN INFORMASI)
# =========================================================
with st.expander("ℹ Penjelasan Singkat Algoritma"):
    st.write("""
    *Genetic Algorithm (GA)* adalah algoritma optimasi berbasis evolusi.  
    Komponen utamanya:
    - Seleksi
    - Crossover
    - Mutasi  
    - Replacement

    *SODGA* adalah mekanisme otomatis untuk memilih metode seleksi terbaik setiap generasi
    menggunakan metode keputusan multikriteria *TOPSIS*.  
    Metode seleksi yang dipertimbangkan:
    - *RWS* (Roulette Wheel Selection)
    - *SUS* (Stochastic Universal Sampling)
    - *TS* (Tournament Selection)
    - *RS* (Rank Selection)

    TOPSIS memilih metode terbaik berdasarkan dinamika nilai fitness populasi.
    
    Keterangan CVRP :
    - **Depot** = node 0.
    - **Customers** = node 1..N.
    - **Chromosome**: list yang berisi customers dan separator '0' (depot) sebagai pemisah rute.
      Contoh: `[3,1,0,5,2,4]` → Vehicle1: `3 -> 1` ; Vehicle2: `5 -> 2 -> 4`.
    - **Fitness**: 1 / (total_distance + penalty * overload). Penalti besar mencegah overload.
    - **SODGA**: memilih metode seleksi terbaik tiap generasi menggunakan TOPSIS.
    """)

# ---------------------------
# Sidebar: parameters
# ---------------------------
st.sidebar.header("⚙ CVRP Parameters")

num_customers = st.sidebar.number_input("Jumlah Customer (tidak termasuk depot)", 2, 60, 8)
num_vehicles = st.sidebar.number_input("Jumlah Kendaraan", 1, 20, 3)
capacity = st.sidebar.number_input("Kapasitas per kendaraan", 1, 500, 30)

population_size = st.sidebar.number_input("Population Size", 10, 500, 60)
generations = st.sidebar.number_input("Generations", 5, 500, 150)

crossover_rate = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.9)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.2)

st.sidebar.markdown("---")
demand_mode = st.sidebar.radio("Demand input", ("Random", "Manual CSV"))
if demand_mode == "Random":
    min_d = st.sidebar.number_input("Min demand", 0, 100, 1)
    max_d = st.sidebar.number_input("Max demand", 0, 100, 10)
else:
    demand_text = st.sidebar.text_area("Masukkan demand untuk customers 1..N (pisahkan koma)", ", ".join(["5"] * num_customers))

st.sidebar.markdown("---")
show_animation = st.sidebar.checkbox("Tampilkan Animasi Evolusi Rute (best-so-far)", True)
anim_speed = st.sidebar.slider("Kecepatan animasi (detik/frame)", 0.01, 0.5, 0.08)

run_button = st.sidebar.button("▶ Jalankan GA (CVRP)")

# ---------------------------
# Helper functions
# ---------------------------
def generate_distance_matrix_cvrp(num_customers):
    n = num_customers + 1  # include depot at index 0
    coords = np.random.rand(n, 2) * 100
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
    return matrix, coords

def plot_cvrp_routes(coords, route, num_vehicles, depot=0):
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(num_vehicles)]
    vehicle_idx = 0
    cur = []
    for token in route + [0]:
        if token == 0:
            if len(cur) > 0:
                xs = [coords[depot][0]] + [coords[n][0] for n in cur] + [coords[depot][0]]
                ys = [coords[depot][1]] + [coords[n][1] for n in cur] + [coords[depot][1]]
                ax.plot(xs, ys, marker='o', color=colors[vehicle_idx % num_vehicles], linewidth=2, label=f'Veh {vehicle_idx+1}')
                vehicle_idx += 1
            cur = []
        else:
            cur.append(token)
    # draw nodes
    ax.scatter(coords[1:, 0], coords[1:, 1], s=35, color='black', zorder=3)
    for i in range(1, len(coords)):
        ax.text(coords[i,0] + 0.5, coords[i,1] + 0.5, str(i), color='black', fontsize=9)
    # depot
    ax.scatter(coords[0:1,0], coords[0:1,1], s=80, color='red', marker='s', zorder=4)
    ax.text(coords[0,0] + 0.5, coords[0,1] + 0.5, "Depot(0)", color='red', fontsize=10)
    ax.set_title("CVRP Routes (colors = vehicles)")
    ax.legend(loc='upper right', fontsize='small')
    return fig

# ---------------------------
# Main: run GA when button pressed
# ---------------------------
if run_button:
    # -----------------------
    # prepare demand
    # -----------------------
    if demand_mode == "Random":
        if min_d > max_d:
            st.error("Min demand tidak boleh lebih besar dari Max demand")
            st.stop()
        demand = [0] + [random.randint(min_d, max_d) for _ in range(num_customers)]
    else:
        try:
            arr = [int(x.strip()) for x in demand_text.split(",") if x.strip() != ""]
            if len(arr) < num_customers:
                st.error(f"Jumlah demand ({len(arr)}) kurang dari jumlah customer ({num_customers}).")
                st.stop()
            demand = [0] + arr[:num_customers]
        except Exception:
            st.error("Format demand tidak benar — gunakan angka dipisah koma.")
            st.stop()

    # -----------------------
    # distance matrix & show heatmap
    # -----------------------
    distance_matrix, coords = generate_distance_matrix_cvrp(num_customers)

    st.subheader("📍 Heatmap Jarak (termasuk depot index 0)")
    fig_hm, ax_hm = plt.subplots(figsize=(6, 5))
    sns.heatmap(distance_matrix, cmap="viridis", ax=ax_hm)
    ax_hm.set_title("Heatmap Distance Matrix")
    st.pyplot(fig_hm)

    # -----------------------
    # initialize population
    # -----------------------
    st.markdown("<div class='card'><b>🚀 Inisialisasi populasi...</b></div>", unsafe_allow_html=True)
    population = init_cvrp_population(population_size, num_customers, num_vehicles)

    # Set up logging + progress UI
    log_container = st.empty()
    # We'll keep the log text in a variable and update textarea content
    log_text = ""
    def append_log(line):
        global log_text
        timestamp = time.strftime("%H:%M:%S")
        log_text += f"[{timestamp}] {line}\n"
        # keep textarea scrollable and readable
        log_container.text_area("📝 Log Proses GA (Realtime)", value=log_text, height=300)

    append_log(f"Populasi diinisialisasi: {len(population)} individu, customers={num_customers}, vehicles={num_vehicles}, capacity={capacity}")
    append_log(f"Demand sample (1..N): {demand[1:]}")

    # GA bookkeeping
    best_fitness_per_gen = []
    best_route = None
    best_fit_val = -1.0
    topsis_log = []
    route_frames = []

    # progress bar + status
    progress = st.progress(0)
    status = st.empty()

    # Main GA loop
    for gen in range(generations):
        # compute fitness
        fitness_values = np.array([cvrp_fitness(ind, distance_matrix, demand, capacity) for ind in population])

        # get generation best
        gen_best_idx = int(np.argmax(fitness_values))
        gen_best = population[gen_best_idx]
        gen_best_fit = float(fitness_values[gen_best_idx])

        # update global best
        if gen_best_fit > best_fit_val:
            best_fit_val = gen_best_fit
            best_route = gen_best.copy()
            append_log(f"Generasi {gen+1}: Best baru ditemukan (fitness={best_fit_val:.6f}).")

        best_fitness_per_gen.append(best_fit_val)
        route_frames.append(best_route.copy())

        # SODGA selection decision
        best_method, scores = select_best_method(fitness_values)
        try:
            rws_score, sus_score, ts_score, rs_score = [float(s) for s in scores]
        except Exception:
            rws_score = sus_score = ts_score = rs_score = 0.0

        topsis_log.append({
            "Gen": gen + 1,
            "RWS": rws_score, "SUS": sus_score, "TS": ts_score, "RS": rs_score, "Best": best_method
        })

        # log SODGA decision & stats
        append_log(f"Generasi {gen+1}/{generations} — MeanFit={np.mean(fitness_values):.6f}, Range={np.max(fitness_values)-np.min(fitness_values):.6f}, Std={np.std(fitness_values):.6f}")
        append_log(f"  TOPSIS scores → RWS={rws_score:.4f}, SUS={sus_score:.4f}, TS={ts_score:.4f}, RS={rs_score:.4f} → dipilih: {best_method}")

        # Selection (build parent pool)
        parents = []
        for _ in range(population_size):
            if best_method == "RWS":
                parents.append(rws_selection(population, fitness_values))
            elif best_method == "SUS":
                parents.append(sus_selection(population, fitness_values))
            elif best_method == "TS":
                parents.append(tournament_selection(population, fitness_values))
            else:
                parents.append(rank_selection(population, fitness_values))

        append_log(f"  Seleksi: dibentuk pool parent dengan metode {best_method} (size={len(parents)})")

        # Crossover
        offspring = []
        crossover_count = 0
        for i in range(0, population_size, 2):
            p1 = parents[i]
            p2 = parents[(i+1) % population_size]
            if random.random() < crossover_rate:
                c1 = ox_crossover_cvrp(p1, p2, num_customers, num_vehicles)
                c2 = ox_crossover_cvrp(p2, p1, num_customers, num_vehicles)
                crossover_count += 1
            else:
                c1 = p1.copy()
                c2 = p2.copy()
            offspring.append(c1)
            offspring.append(c2)
        append_log(f"  Crossover: dilakukan pada {crossover_count} pasangan (rate={crossover_rate})")

        # Mutation
        mutation_count = 0
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                offspring[i] = inversion_mutation_cvrp(offspring[i], num_customers, num_vehicles, mut_rate=1.0)
                mutation_count += 1
        append_log(f"  Mutasi: diterapkan pada {mutation_count} individu (rate={mutation_rate})")

        # Repair offspring & replacement
        offspring = [repair_route(o, num_customers, num_vehicles) for o in offspring]
        population = replace_population_cvrp(population, offspring, distance_matrix, demand, capacity)
        append_log(f"  Replacement: populasi diganti dengan generasi baru.")

        # update progress + status for UI
        progress.progress(int((gen + 1) / generations * 100))
        status.info(f"Generasi {gen+1}/{generations} — Best fitness (global): {best_fit_val:.6f} — Selected: {best_method}")

    # GA loop finished
    progress.empty()
    status.success("Proses GA selesai ✅")
    append_log("GA selesai. Menampilkan hasil akhir.")

    # ---------------------------
    # OUTPUT: results visualization & downloads
    # ---------------------------
    st.markdown("---")
    st.header("🏁 Hasil Akhir CVRP")

    st.subheader("🔗 Best Route (chromosome)")
    st.write(best_route)

    # plot routes
    fig_routes = plot_cvrp_routes(coords, best_route, num_vehicles)
    st.pyplot(fig_routes)

    # best distance (approx via fitness)
    st.subheader("📏 Best Distance (approx.)")
    best_distance = (1.0 / best_fit_val) if best_fit_val > 0 else float("inf")
    st.success(f"{best_distance:.4f}")

    # fitness chart
    st.subheader("📈 Fitness Per Generasi (best-so-far)")
    st.line_chart(best_fitness_per_gen)

    # topsis table
    st.subheader("📊 Skor TOPSIS per Generasi (sample terakhir)")
    topsis_df = pd.DataFrame(topsis_log)
    st.dataframe(topsis_df.tail(10))

    # ---------------------------
    # Animation with status indicator
    # ---------------------------
    if show_animation and len(route_frames) > 0:
        st.subheader("🎬 Animasi Evolusi Rute (best-so-far tiap generasi)")
        anim_status = st.empty()
        anim_placeholder = st.empty()
        anim_status.info(f"Menjalankan animasi — {len(route_frames)} frame akan ditampilkan.")
        for i, rf in enumerate(route_frames):
            anim_status.info(f"🔄 Animasi: menampilkan frame {i+1}/{len(route_frames)}")
            figf = plot_cvrp_routes(coords, rf, num_vehicles)
            anim_placeholder.pyplot(figf)
            time.sleep(anim_speed)
        anim_placeholder.empty()
        anim_status.success("✔ Animasi selesai.")

    # ---------------------------
    # Save results (JSON + CSV)
    # ---------------------------
    st.markdown("---")
    st.subheader("💾 Save Results")

    results = {
        "best_route": best_route,
        "best_distance": float(best_distance),
        "fitness_per_generation": best_fitness_per_gen,
        "topsis_log": topsis_log,
        "params": {
            "num_customers": num_customers,
            "num_vehicles": num_vehicles,
            "capacity": capacity,
            "population_size": population_size,
            "generations": generations,
            "crossover_rate": float(crossover_rate),
            "mutation_rate": float(mutation_rate)
        },
        "demand": demand
    }

    st.download_button("📥 Download JSON", json.dumps(results, indent=2), "cvrp_results.json", "application/json")
    st.download_button("📥 Download TOPSIS CSV", topsis_df.to_csv(index=False), "topsis_scores.csv", "text/csv")

    # ---------------------------
    # Simple Markdown report with embedded fitness PNG
    # ---------------------------
    st.subheader("📝 Generate Simple Report (Markdown)")

    figf2, ax2 = plt.subplots()
    ax2.plot(best_fitness_per_gen)
    ax2.set_title("Fitness (best-so-far)")
    buf = BytesIO()
    figf2.savefig(buf, format="png")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()

    report_md = f"""
# CVRP Report

**Parameters**
- customers: {num_customers}
- vehicles: {num_vehicles}
- capacity: {capacity}
- population: {population_size}
- generations: {generations}
- crossover_rate: {crossover_rate}
- mutation_rate: {mutation_rate}

**Best distance:** {best_distance:.4f}  
**Best route:** {best_route}

![fitness](data:image/png;base64,{b64})
"""
    st.download_button("📥 Download Report (MD)", report_md, "cvrp_report.md", "text/markdown")

else:
    st.info("Tekan tombol ▶ Jalankan GA (CVRP) di sidebar untuk memulai.")
