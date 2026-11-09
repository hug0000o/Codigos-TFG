import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. FUNCIONES DE GEOMETRÍA Y TOPOLOGÍA
# ==========================================
@njit(fastmath=True)
def inside(x, y, L, topology):
    if topology == 0:
        if 0 <= x <= L and 0 <= y <= L: return True
        return False
    if topology == 1:    
        if L <= x <= 2*L and 0 <= y <= 4*L: return True
        if 0 <= x <= 3*L and L <= y <= 2*L: return True
    return False

@njit(fastmath=True)
def PBC(old_x, old_y, new_x, new_y, L, topology):
    if topology == 0: return new_x % L, new_y % L
    if topology == 1:
        if inside(new_x, new_y, L, topology): return new_x, new_y
        if old_y < L: 
            if new_y < 0: return new_x, new_y + 4*L               
            if new_x < L: return new_y, 2*L - new_x               
            if new_x > 2*L: return 3*L - new_y, new_x - L         
        elif old_y > 3*L: 
            if new_y > 4*L: return new_x, new_y - 4*L             
            if new_x < L: return L - new_x, 5*L - new_y           
            if new_x > 2*L: return 5*L - new_x, 5*L - new_y       
        elif old_x < L: 
            if new_y < L: return 2*L - new_y, new_x               
            if new_x < 0: return L - new_x, 5*L - new_y           
            if new_y > 2*L: return new_y - L, 3*L - new_x         
        elif old_x > 2*L: 
            if new_y < L: return L + new_y, 3*L - new_x           
            if new_x > 3*L: return 5*L - new_x, 5*L - new_y       
            if new_y > 2*L: return 4*L - new_y, new_x             
        else: 
            if 2*L <= old_y <= 3*L: 
                if new_x < L: return 3*L - new_y, L + new_x       
                if new_x > 2*L: return new_y, 4*L - new_x         
        return new_x, new_y

@njit(fastmath=True)
def get_ghosts(x, y, L, threshold, topology):
    ghosts = np.full((14, 2), -100.0) 
    idx = 0
    if topology == 0:
        if x < threshold: ghosts[idx] = (x + L, y); idx+=1
        elif x > L - threshold: ghosts[idx] = (x - L, y); idx+=1
        if y < threshold: ghosts[idx] = (x, y + L); idx+=1
        elif y > L - threshold: ghosts[idx] = (x, y - L); idx+=1
        if x < threshold and y < threshold: ghosts[idx] = (x + L, y + L); idx+=1
        elif x > L - threshold and y < threshold: ghosts[idx] = (x - L, y + L); idx+=1
        elif x < threshold and y > L - threshold: ghosts[idx] = (x + L, y - L); idx+=1
        elif x > L - threshold and y > L - threshold: ghosts[idx] = (x - L, y - L); idx+=1
        return ghosts
    if topology == 1:
        if y < threshold and L <= x <= 2*L: ghosts[idx] = (x, y + 4*L); idx+=1
        if y > 4*L - threshold and L <= x <= 2*L: ghosts[idx] = (x, y - 4*L); idx+=1
        if x < L + threshold and x > L and y < L: ghosts[idx] = (y, 2*L - x); idx+=1
        if y < L + threshold and y > L and x < L: ghosts[idx] = (2*L - y, x); idx+=1
        if x > 2*L - threshold and x < 2*L and y < L: ghosts[idx] = (3*L - y, x - L); idx+=1
        if y < L + threshold and y > L and x > 2*L: ghosts[idx] = (L + y, 3*L - x); idx+=1
        if x < L + threshold and x > L and 2*L < y < 3*L: ghosts[idx] = (3*L - y, L + x); idx+=1
        if y > 2*L - threshold and y < 2*L and x < L: ghosts[idx] = (y - L, 3*L - x); idx+=1
        if x > 2*L - threshold and x < 2*L and 2*L < y < 3*L: ghosts[idx] = (y, 4*L - x); idx+=1
        if y > 2*L - threshold and y < 2*L and x > 2*L: ghosts[idx] = (4*L - y, x); idx+=1
        if x < L + threshold and x > L and y > 3*L: ghosts[idx] = (L - x, 5*L - y); idx+=1
        if x < threshold and L < y < 2*L: ghosts[idx] = (L - x, 5*L - y); idx+=1 
        if x > 2*L - threshold and x < 2*L and y > 3*L: ghosts[idx] = (5*L - x, 5*L - y); idx+=1
        if x > 3*L - threshold and L < y < 2*L: ghosts[idx] = (5*L - x, 5*L - y); idx+=1 
    return ghosts

@njit(fastmath=True)
def check_grid(px, py, x, y, D, head, next_node, ncx, ncy, ignore_idx):
    cx, cy = int(px/D), int(py/D)
    for dcx in (-1, 0, 1):
        for dcy in (-1, 0, 1):
            ccx = cx + dcx
            ccy = cy + dcy
            if 0 <= ccx < ncx and 0 <= ccy < ncy:
                j = head[ccx + ccy * ncx]
                while j != -1:
                    if j != ignore_idx:
                        if (px - x[j])**2 + (py - y[j])**2 < D*D:
                            return True
                    j = next_node[j]
    return False

@njit(fastmath=True)
def check_overlap(px, py, x, y, D, head, next_node, ncx, ncy, L, topology, ignore_idx=-1):
    if check_grid(px, py, x, y, D, head, next_node, ncx, ncy, ignore_idx): return True
    ghosts = get_ghosts(px, py, L, D, topology)
    for i in range(8):
        gx, gy = ghosts[i]
        if gx != -100.0: 
            if check_grid(gx, gy, x, y, D, head, next_node, ncx, ncy, ignore_idx): 
                return True
    return False

# ==========================================
# 2. MOTOR DE MONTE CARLO
# ==========================================
@njit(fastmath=True)
def init_system(x, y, L, D_eff, head, next_node, ncx, ncy, topology):
    head[:] = -1
    count = 0
    N_total = len(x)
    max_x = L if topology == 0 else 3*L
    max_y = L if topology == 0 else 4*L
    
    while count < N_total:
        px = np.random.uniform(0, max_x)
        py = np.random.uniform(0, max_y)
        
        if not inside(px, py, L, topology): continue
        if not check_overlap(px, py, x, y, D_eff, head, next_node, ncx, ncy, L, topology, -1):
            x[count], y[count] = px, py
            idx = int(px/D_eff) + int(py/D_eff) * ncx
            next_node[count] = head[idx]
            head[idx] = count
            count += 1
    return count

@njit(fastmath=True)
def mc_step(x, y, L, D_eff, max_disp, head, next_node, ncx, ncy, topology):
    N_total = len(x)
    accepts = 0
    for i in range(N_total):
        old_x, old_y = x[i], y[i]
        dx = (np.random.rand() - 0.5) * 2 * max_disp
        dy = (np.random.rand() - 0.5) * 2 * max_disp
        new_x, new_y = PBC(old_x, old_y, old_x + dx, old_y + dy, L, topology)
        
        if not inside(new_x, new_y, L, topology): continue 
            
        if not check_overlap(new_x, new_y, x, y, D_eff, head, next_node, ncx, ncy, L, topology, i):
            x[i], y[i] = new_x, new_y
            accepts += 1
            
            old_idx = int(old_x/D_eff) + int(old_y/D_eff) * ncx
            new_idx = int(new_x/D_eff) + int(new_y/D_eff) * ncx
            if old_idx != new_idx:
                prev, curr = -1, head[old_idx]
                while curr != -1:
                    if curr == i:
                        if prev == -1: head[old_idx] = next_node[curr]
                        else: next_node[prev] = next_node[curr]
                        break
                    prev, curr = curr, next_node[curr]
                next_node[i] = head[new_idx]
                head[new_idx] = i
    return accepts / N_total

@njit(fastmath=True)
def compression(x, y, L, D_eff, shrink_factor, head, next_node, ncx, ncy, topology):
    old_x = x.copy()
    old_y = y.copy()
    new_L = L * shrink_factor
    scale_ratio = new_L / L
    
    for i in range(len(x)):
        x[i] = x[i] * scale_ratio
        y[i] = y[i] * scale_ratio
        
    head[:] = -1
    for i in range(len(x)):
        idx = int(x[i]/D_eff) + int(y[i]/D_eff) * ncx
        next_node[i] = head[idx]
        head[idx] = i
        
    overlap = False
    for i in range(len(x)):
        if check_overlap(x[i], y[i], x, y, D_eff, head, next_node, ncx, ncy, new_L, topology, i):
            overlap = True
            break
            
    if overlap:
        for i in range(len(x)):
            x[i] = old_x[i]
            y[i] = old_y[i]
            
        head[:] = -1
        for i in range(len(x)):
            idx = int(x[i]/D_eff) + int(y[i]/D_eff) * ncx
            next_node[i] = head[idx]
            head[idx] = i
            
        return L, False 
        
    return new_L, True  

# ==========================================
# 3. ENCAPSULACIÓN Y ANÁLISIS
# ==========================================
def correr_simulacion(N, MC_STEPS, TOPOLOGY=1):
    """Ejecuta una simulación completa y devuelve el tiempo transcurrido."""
    D = 1.0                                   
    R = D / 2.0                               
    phi_init = 0.1                            
    area_particulas = N * np.pi * R**2
    area_total = area_particulas / phi_init

    if TOPOLOGY == 0:
        L = np.sqrt(area_total)                 
    else:
        L = np.sqrt(area_total / 6.0)            
        
    MAX_DISP = D / 10.0                             
    GAMMA = D / 40.0                           
    D_EFF = D + GAMMA                         

    SHRINK_FACTOR = 0.99                      
    TOL_VOL = 2.0e-12                         

    if TOPOLOGY == 0:
        DOM_W, DOM_H = L, L                              
    else:
        DOM_W, DOM_H = 3*L, 4*L                             

    NCX = int(DOM_W / D_EFF) + 1              
    NCY = int(DOM_H / D_EFF) + 1              

    x_pos = np.zeros(N, dtype=np.float64)
    y_pos = np.zeros(N, dtype=np.float64)
    head = np.full(NCX * NCY, -1, dtype=np.int32)
    next_node = np.full(N, -1, dtype=np.int32)
    
    t_start = time.time()
    init_system(x_pos, y_pos, L, D_EFF, head, next_node, NCX, NCY, TOPOLOGY)
    
    def get_volume(L_val, top):
        return L_val**2 if top == 0 else 6 * L_val**2
        
    V_history = np.zeros(3, dtype=np.float64)
    V_history[0] = get_volume(L, TOPOLOGY)
    V_history[1] = V_history[0]
    
    acc = 1.0 
    
    for step in range(MC_STEPS):
        if step > 0:
            if acc > 0.50:
                MAX_DISP = min(MAX_DISP * 1.05, D / 2.0)
            elif acc < 0.30:
                MAX_DISP = max(MAX_DISP * 0.95, D / 200.0) 
                
            if acc > 0.8: SHRINK_FACTOR = 0.9999    
            elif acc > 0.6: SHRINK_FACTOR = 0.9999   
            else: SHRINK_FACTOR = 0.9999  

        L, accepted = compression(x_pos, y_pos, L, D_EFF, SHRINK_FACTOR, head, next_node, NCX, NCY, TOPOLOGY)
        acc = mc_step(x_pos, y_pos, L, D_EFF, MAX_DISP, head, next_node, NCX, NCY, TOPOLOGY)
        
        V_history[2] = get_volume(L, TOPOLOGY)
        
        if accepted:
            delta_V = V_history[0] - V_history[2] 
            if 0 < delta_V < TOL_VOL:
                break
                
        V_history[0] = V_history[1]
        V_history[1] = V_history[2]

    return time.time() - t_start

if __name__ == "__main__":
    # ==========================================
    # SELECTOR DE MODO DE ANÁLISIS
    # Cambia a "PASOS" o "PARTICULAS" según lo que quieras generar
    # ==========================================
    MODO_ANALISIS = "PASOS" 
    
    print("Compilando Numba en vacío (Warm-up)...")
    correr_simulacion(N=100, MC_STEPS=100) # Evita que el primer tiempo incluya la compilación de Numba
    
    tiempos = []

    if MODO_ANALISIS == "PASOS":
        particulas_fijas = 500
        vector_pasos = [10000, 50000, 100000, 250000, 500000]
        
        print(f"Iniciando análisis por PASOS MC (N fijo = {particulas_fijas})")
        for pasos in vector_pasos:
            print(f"Simulando {pasos} pasos...")
            t = correr_simulacion(N=particulas_fijas, MC_STEPS=pasos)
            tiempos.append(t)
            print(f" -> Tiempo: {t:.2f} s")
            
        plt.figure(figsize=(8, 5))
        plt.plot(vector_pasos, tiempos, marker='o', linestyle='-', color='b')
        plt.title(f"Complejidad Temporal vs Pasos de MC (N={particulas_fijas})")
        plt.xlabel("Número de Pasos (MC_STEPS)")
        plt.ylabel("Tiempo de Ejecución (s)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    elif MODO_ANALISIS == "PARTICULAS":
        pasos_fijos = 50000
        vector_particulas = [500, 1000, 2000, 5000, 10000]
        
        print(f"Iniciando análisis por NÚMERO DE PARTÍCULAS (Pasos fijos = {pasos_fijos})")
        for particulas in vector_particulas:
            print(f"Simulando {particulas} partículas...")
            t = correr_simulacion(N=particulas, MC_STEPS=pasos_fijos)
            tiempos.append(t)
            print(f" -> Tiempo: {t:.2f} s")

        plt.figure(figsize=(8, 5))
        plt.plot(vector_particulas, tiempos, marker='s', linestyle='-', color='r')
        plt.title(f"Complejidad Temporal vs Número de Partículas (Pasos={pasos_fijos})")
        plt.xlabel("Número de Partículas (N)")
        plt.ylabel("Tiempo de Ejecución (s)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()