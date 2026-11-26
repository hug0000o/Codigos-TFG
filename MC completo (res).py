import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import time
import modulo_tfg as mtfg
import math





# parámetros

TOPOLOGY = 0                              # 0: cuadrado simple, 1: topología del cubo
D = 1.0                                   # Diámetro de las partículas
R = D / 2.0                               # Radio de las partículas
N = 100                                  # Número de partículas

phi_init = 0.1                            # Fracción de área inicial
area_particulas = N * np.pi * R**2
area_total = area_particulas / phi_init

if TOPOLOGY == 0:
    L = np.sqrt(area_total)                 
else:
    L = np.sqrt(area_total / 6.0)            
    
MAX_DISP = D / 10.0                       # desplazamiento máximo       
GAMMA = 1e-12                              # esfera de influencia para proteger contra solapamientos numéricos pequeños y evitar imprecisiones numéricas al usar el fastmath de numba
D_EFF = D + GAMMA                         # diámetro efectivo

SHRINK_FACTOR = 0.99999                      # factor de contracción de la caja inicial
TOL_VOL = 2.0e-12                         # criterio de parada 

if TOPOLOGY == 0:
    DOM_W = L                              # ancho del dominio de simulación 
    DOM_H = L                              # alto del dominio de simulación
else:
    DOM_W = 3*L                             
    DOM_H = 4*L                             

NCX = int(DOM_W / D_EFF) + 1              # número de celdas en X
NCY = int(DOM_H / D_EFF) + 1              # número de celdas en Y


MC_STEPS = 2000000                         # pasos del MC    
imp_steps = MC_STEPS // 20                 # pasos entre impresiones de progreso





# simulación
if __name__ == "__main__":
    print(f"Inicializando...")
    
    x_pos = np.zeros(N, dtype=np.float64)
    y_pos = np.zeros(N, dtype=np.float64)
    head = np.full(NCX * NCY, -1, dtype=np.int32)
    next_node = np.full(N, -1, dtype=np.int32)
    
    t_start = time.time()
    mtfg.init_system(x_pos, y_pos, L, D_EFF, head, next_node, NCX, NCY, TOPOLOGY)
    print(f"Inicialización RSA completada. Comenzando con {MC_STEPS} pasos de Monte Carlo...")
    
        
    V_history = np.zeros(3, dtype=np.float64)
    V_history[0] = mtfg.get_area(L, TOPOLOGY)
    V_history[1] = V_history[0]
    
    acc = 1.0 
    COMPRESS_EVERY = 10
    
    for step in range(MC_STEPS):
        
        if step > 0:
            if acc > 0.35:
                MAX_DISP = min(MAX_DISP * 1.05, D / 2.0)
            elif acc < 0.25:
                MAX_DISP = max(MAX_DISP * 0.95, D * 1e-6)
                
                
        accepted = False
        if step % COMPRESS_EVERY == 0:
            L, accepted = mtfg.compression(x_pos, y_pos, L, D_EFF, SHRINK_FACTOR, head, next_node, NCX, NCY, TOPOLOGY)

        acc = mtfg.mc_step(x_pos, y_pos, L, D_EFF, MAX_DISP, head, next_node, NCX, NCY, TOPOLOGY)
          
        
        # parada
        V_history[2] = mtfg.get_area(L, TOPOLOGY)
        
        if accepted:
            delta_V = V_history[0] - V_history[2] 
            if 0 < delta_V < TOL_VOL:
                print(f"\n--- JAMMING ESTRICTO ALCANZADO ---")
                print(f"Paso: {step} | Delta V: {delta_V:.3e} < {TOL_VOL}")
                break
                
        V_history[0] = V_history[1]
        V_history[1] = V_history[2]
        
        if (step + 1) % imp_steps == 0:
            elapsed = time.time() - t_start
            area_particulas = N * np.pi * R**2
            phi_actual = area_particulas / mtfg.get_area(L, TOPOLOGY)
            print(f"Progreso: {step+1}/{MC_STEPS} | ϕ: {phi_actual:.4f} | L: {L:.4f} | Acc: {acc*100:.1f}% | Disp: {MAX_DISP:.4f} | T: {elapsed:.1f} s")
    
    #resultado
    t_end = time.time()
    phi_final = area_particulas / mtfg.get_area(L, TOPOLOGY)
    
    print(f"\nSimulación finalizada:")
    print(f"\nNúmero de Partículas: {N}")
    print(f"Tiempo Total: {t_end - t_start:.2f} segundos")
    print(f"L Final: {L:.5f}")
    print(f"Fracción de Área Final (ϕ): {phi_final:.4f}")
    

    print(f"\nGenerando imagen del estado final...")
    fig, ax = plt.subplots(figsize=(6, 8) if TOPOLOGY == 1 else (6, 6))
    ax.set_aspect('equal')
    
    if TOPOLOGY == 0:
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        boundary_coords = [(0,0), (L,0), (L,L), (0,L)]
    else:
        ax.set_xlim(0, 3*L)
        ax.set_ylim(0, 4*L)
        boundary_coords = [(L,0), (2*L,0), (2*L,L), (3*L,L), (3*L,2*L), (2*L,2*L), 
                           (2*L,4*L), (L,4*L), (L,2*L), (0,2*L), (0,L), (L,L)]     
        
    boundary_patch = Polygon(boundary_coords, closed=True, fill=False, edgecolor='black', lw=1.5, zorder=2)
    ax.add_patch(boundary_patch)
    
    for i in range(N):
        c = Circle((x_pos[i], y_pos[i]), R, color='crimson', alpha=0.9, zorder=3)
        c.set_clip_path(boundary_patch)
        ax.add_patch(c)
        
        ghosts = mtfg.get_ghosts(x_pos[i], y_pos[i], L, R, TOPOLOGY) 
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