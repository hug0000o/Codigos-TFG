import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from numba import njit
import time
import modulo_motor as mm
import modulo_analisis as ma

# parámetros

TOPOLOGY = 0                              # 0: cuadrado simple, 1: topología del cubo
D = 1.0                                   # Diámetro de las partículas (sigma)
R = D / 2.0                               # Radio de las partículas
N = 100                                   # Número de partículas

phi_init = 0.1                            # Fracción de área inicial
area_particulas = N * np.pi * R**2
area_total = area_particulas / phi_init

if TOPOLOGY == 0:
    L = np.sqrt(area_total)
elif TOPOLOGY == 1:
    L = np.sqrt(area_total / 6.0)

MAX_DISP = D / 10.0                       # Desplazamiento máximo inicial
GAMMA = 1e-12                             # Margen numérico para fastmath
D_EFF = D + GAMMA                         # Diámetro efectivo

# Compresión y parada 
COMPRESS_EVERY = 10                       # Comprimir 1 vez cada X barridos MC
DELTA_S_INIT = 1e-3                       # Salto inicial
TOL_SHRINK = 1e-12                        # Criterio de parada: precisión final de compresión
REJECTS_TO_HALVE = 1000                   # Rechazos seguidos para dividir delta_s a la mitad

if TOPOLOGY == 0:
    DOM_W = L
    DOM_H = L
elif TOPOLOGY == 1:
    DOM_W = 3 * L
    DOM_H = 4 * L

NCX = int(DOM_W / D_EFF) + 1              # Número de celdas en X
NCY = int(DOM_H / D_EFF) + 1              # Número de celdas en Y

MC_STEPS = 3000000                        # Techo máximo de pasos MC
imp_steps = 50000                         # Pasos entre impresiones de progreso


DR = 0.01 * D                             # Ancho del bin para g(r)
N_THERM_SAMPLE = 300                      # Pasos MC de relajación a phi fija antes de medir
N_CNF_SAMPLE = 300                        # Configuraciones acumuladas por punto de presión
SAMPLE_EVERY = 5                          # Pasos MC entre configuraciones muestreadas
N_PUNTOS_FIT = 5                          # Bines por encima de D para extrapolar g(D+)

             


# simulación

if __name__ == "__main__":
    print("Inicializando...")

    x_pos = np.zeros(N, dtype=np.float64)
    y_pos = np.zeros(N, dtype=np.float64)
    head = np.full(NCX * NCY, -1, dtype=np.int32)
    next_node = np.full(N, -1, dtype=np.int32)

    t_start = time.time()
    mm.init_system(x_pos, y_pos, L, D_EFF, head, next_node, NCX, NCY, TOPOLOGY)
    print(f"Inicialización RSA completada. Comenzando simulación (máx. {MC_STEPS} pasos MC)...")

    
    phis_eos = []
    Z_eos = []
    P_eos = []

    
    phi_0, Z_0, P_0, gc_0, r_init, g_init = ma.medir_estado(x_pos, y_pos, L, D_EFF, MAX_DISP, head, next_node, NCX, NCY, TOPOLOGY, D, area_particulas)
    phis_eos.append(phi_0)
    Z_eos.append(Z_0)
    P_eos.append(P_0)
    next_phi_sample = phi_0 + ma.get_delta_phi(phi_0)
    print(f"[Muestra inicial] ϕ: {phi_0:.4f} | g(D+): {gc_0:.3f} | βPa₀: {P_0:.3f}")

    acc = 1.0
    delta_s = DELTA_S_INIT
    reject_streak = 0

    for step in range(MC_STEPS):

        if step > 0:
            if acc > 0.35:
                MAX_DISP = min(MAX_DISP * 1.05, D / 2.0)
            elif acc < 0.25:
                MAX_DISP = max(MAX_DISP * 0.95, D * 1e-13)

        if step % COMPRESS_EVERY == 0:
            shrink = 1.0 - delta_s
            L, accepted = mm.compression(
                x_pos, y_pos, L, D_EFF, shrink, head, next_node, NCX, NCY, TOPOLOGY
            )

            if accepted:
                reject_streak = 0
                phi_actual = area_particulas / mm.get_area(L, TOPOLOGY)

                if phi_actual >= next_phi_sample:
                    phi_m, Z_m, P_m, gc_m, _, _ = ma.medir_estado(x_pos, y_pos, L, D_EFF, MAX_DISP, head, next_node, NCX, NCY, TOPOLOGY, D, area_particulas)
                    phis_eos.append(phi_m)
                    Z_eos.append(Z_m)
                    P_eos.append(P_m)
                    next_phi_sample = phi_m + ma.get_delta_phi(phi_m)
            else:
                reject_streak += 1
                if reject_streak >= REJECTS_TO_HALVE:
                    delta_s *= 0.5
                    reject_streak = 0
                    if delta_s < TOL_SHRINK:
                        phi_actual = area_particulas / mm.get_area(L, TOPOLOGY)
                        print(f"\nJamming alcanzado:")
                        print(f"Paso: {step+1} | delta_s: {delta_s:.2e} < {TOL_SHRINK:.1e}")
                        break

        acc = mm.mc_step(x_pos, y_pos, L, D_EFF, MAX_DISP, head, next_node, NCX, NCY, TOPOLOGY)

        if (step + 1) % imp_steps == 0:
            elapsed = time.time() - t_start
            phi_actual = area_particulas / mm.get_area(L, TOPOLOGY)
            print(f"Progreso: {step+1}/{MC_STEPS} | ϕ: {phi_actual:.4f} | L: {L:.4f} | Acc: {acc*100:.1f}% | Disp: {MAX_DISP:.2e} | dS: {delta_s:.1e} | T: {elapsed:.1f} s")


    phi_final, Z_final, P_final, gc_final, r_final, g_final = ma.medir_estado(x_pos, y_pos, L, D_EFF, MAX_DISP, head, next_node, NCX, NCY, TOPOLOGY, D, area_particulas)
    phis_eos.append(phi_final)
    Z_eos.append(Z_final)
    P_eos.append(P_final)

    t_end = time.time()
    print(f"\nSimulación finalizada:")
    print(f"Número de Partículas: {N}")
    print(f"Tiempo Total: {t_end - t_start:.2f} segundos")
    print(f"L Final: {L:.5f}")
    print(f"Fracción de Área Final (ϕ): {phi_final:.5f}")
    print(f"Puntos en el mapa de estado: {len(phis_eos)}")
    print(f"g(D+) Final: {gc_final:.3f} | βPa₀ Final: {P_final:.3f}")

    
    fig_eos, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    phis_arr = np.array(phis_eos)
    P_arr = np.array(P_eos)

    phi_teo = np.linspace(phis_arr.min(), phis_arr.max(), 200)
    Z_henderson = (1.0 + phi_teo**2 / 8.0) / (1.0 - phi_teo)**2
    P_henderson = phi_teo * Z_henderson

    ax1.plot(phis_arr, P_arr, 'o-', color='crimson', lw=1.4, ms=3.5, label='Simulación MC')
    ax1.plot(phi_teo, P_henderson, '--', color='black', lw=1.2, label='Henderson (fluido 2D)')
    ax1.set_xlabel(r'Fracción de empaquetamiento $\phi$', fontsize=11)
    ax1.set_ylabel(r'Presión reducida $\beta P a_0$', fontsize=11)
    ax1.set_ylim(0, 45)
    ax1.set_title(f'Mapa de estado (Topología {TOPOLOGY}, N={N})', fontsize=12)
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend()

    ax2.plot(r_init / D, g_init, '-', color='steelblue', lw=1.5, label=f'Inicial ($\\phi={phi_0:.3f}$)')
    ax2.plot(r_final / D, g_final, '-', color='crimson', lw=1.5, label=f'Final ($\\phi={phi_final:.3f}$)')
    ax2.axhline(1.0, color='gray', ls='--', lw=1.0)
    ax2.scatter([1.0, 1.0], [gc_0, gc_final], color=['steelblue', 'crimson'], zorder=5, label=r'$g(D^+)$ extrapolado')
    ax2.set_xlabel(r'$r / D$', fontsize=11)
    ax2.set_ylabel(r'$g(r)$', fontsize=11)
    ax2.set_title(r'Función de correlación radial $g(r)$', fontsize=12)
    ax2.set_xlim(0.8, min(5.0, (0.5 * L) / D))
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    
    print("\nGenerando imagen del estado final...")
    fig, ax = plt.subplots(figsize=(6, 8) if TOPOLOGY == 1 else (6, 6))
    ax.set_aspect('equal')

    if TOPOLOGY == 0:
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        boundary_coords = [(0, 0), (L, 0), (L, L), (0, L)]
    elif TOPOLOGY == 1:
        ax.set_xlim(0, 3 * L)
        ax.set_ylim(0, 4 * L)
        boundary_coords = [
            (L, 0), (2 * L, 0), (2 * L, L), (3 * L, L), (3 * L, 2 * L), (2 * L, 2 * L),
            (2 * L, 4 * L), (L, 4 * L), (L, 2 * L), (0, 2 * L), (0, L), (L, L)]

    boundary_patch = Polygon(boundary_coords, closed=True, fill=False, edgecolor='black', lw=1.5, zorder=2)
    ax.add_patch(boundary_patch)

    for i in range(N):
        c = Circle((x_pos[i], y_pos[i]), R, color='crimson', alpha=0.9, zorder=3)
        c.set_clip_path(boundary_patch)
        ax.add_patch(c)

        ghosts = mm.get_ghosts(x_pos[i], y_pos[i], L, R, TOPOLOGY)
        for j in range(8):
            gx, gy = ghosts[j]
            if gx != -100.0:
                cg = Circle((gx, gy), R, color='crimson', alpha=0.9, zorder=3)
                cg.set_clip_path(boundary_patch)
                ax.add_patch(cg)

    plt.title(f"Estado Final (ϕ = {phi_final:.4f}), N={N}", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()