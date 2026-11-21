import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import matplotlib.animation as animation
import time


# parámetros

TOPOLOGY = 1                              # 0: cuadrado simple, 1: topología del cubo
D = 1                                     # diámetro de las partículas
R = D / 2.0                               # radio de las partículas
N = 1000                                    # número de partículas

phi_init= 0.1                                 # fracción de área inicial
area_particulas = N * np.pi * R**2
area_total= area_particulas / phi_init


if TOPOLOGY == 0:
    L = np.sqrt(area_total)                 
else:
    L = np.sqrt(area_total / 6.0)            
    
  
MAX_DISP = D/10                      # desplazamiento máximo          
GAMMA = 1e-12                          # esfera de influencia para proteger contra solapamientos numéricos pequeños y evitar imprecisiones numéricas al usar el fastmath de numba
D_EFF = D + GAMMA                         


SHRINK_FACTOR = 0.999                      # factor de contracción de la caja inicial
TOL_VOL = 2.0e-12                         # criterio de parada 


if TOPOLOGY == 0:
    DOM_W = L                               # ancho del dominio de simulación
    DOM_H = L                               # alto del dominio de simulación
else:
    DOM_W = 3*L                             
    DOM_H = 4*L                             
    
NCX = int(DOM_W / D_EFF) + 1              # número de celdas en X
NCY = int(DOM_H / D_EFF) + 1              # número de celdas en Y

MC_STEPS = 100000                           # pasos del MC
STEPS_PER_FRAME = 100                       # intervalo de pasos para la animacion
NUM_FRAMES = MC_STEPS // STEPS_PER_FRAME     # frames totales para la animación
imp_steps = 5000                             # intervalo de pasos para imprimir progreso




# funciones

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

#  MC e inicialización

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



"""Primera versión de la función de compresión. Se realiza un intento de densificación y se comprueba si se violan las restricciones de solapamiento. 
Si se violan, se restauran las coordenadas originales y se rechaza el intento."""

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



# simulaicón

if __name__ == "__main__":
    print(f"Inicializando...")
    
    x_pos = np.zeros(N, dtype=np.float64)
    y_pos = np.zeros(N, dtype=np.float64)
    head = np.full(NCX * NCY, -1, dtype=np.int32)
    next_node = np.full(N, -1, dtype=np.int32)
    
    trayectoria = np.zeros((NUM_FRAMES, N, 2), dtype=np.float64)
    L_frames = np.zeros(NUM_FRAMES, dtype=np.float64) 
    
    t_start = time.time()
    init_system(x_pos, y_pos, L, D_EFF, head, next_node, NCX, NCY, TOPOLOGY)
    print(f"Inicialización RSA completada.")
    
    frame_idx = 0
    total_acc = 0.0
    
    V_history = np.zeros(3, dtype=np.float64)
    def get_volume(L_val, top):
        return L_val**2 if top == 0 else 12 * L_val**2
        
    V_history[0] = get_volume(L, TOPOLOGY)
    V_history[1] = V_history[0]
    
    for step in range(MC_STEPS):
        
        if step > 0:
            if acc > 0.50:
                MAX_DISP = min(MAX_DISP * 1.05, D / 2.0)
            elif acc < 0.30:
                MAX_DISP = max(MAX_DISP * 0.95, D / 200.0) 
                
            if acc > 0.6:
                SHRINK_FACTOR = 0.99    
            elif acc > 0.2:
                SHRINK_FACTOR = 0.999   
            else:
                SHRINK_FACTOR = 0.9999  

        L, accepted = compression(x_pos, y_pos, L, D_EFF, SHRINK_FACTOR, head, next_node, NCX, NCY, TOPOLOGY)
        
        acc = mc_step(x_pos, y_pos, L, D_EFF, MAX_DISP, head, next_node, NCX, NCY, TOPOLOGY)
        total_acc += acc
        
        V_history[2] = get_volume(L, TOPOLOGY)
        
        if accepted:
            delta_V = V_history[0] - V_history[2] 
            if 0 < delta_V < TOL_VOL:
                print(f"\n Strict Jamming alcanzado(?)")
                print(f"Paso: {step} | Delta V: {delta_V:.3e} < {TOL_VOL}")
                break
                
        V_history[0] = V_history[1]
        V_history[1] = V_history[2]
        
        if step % STEPS_PER_FRAME == 0 and frame_idx < NUM_FRAMES:
            trayectoria[frame_idx, :, 0] = x_pos
            trayectoria[frame_idx, :, 1] = y_pos
            L_frames[frame_idx] = L
            frame_idx += 1
            
        if (step + 1) % imp_steps == 0:                         
            elapsed = time.time() - t_start
            print(f"Progreso: {step+1}/{MC_STEPS} | L_actual: {L:.4f} | Aceptación: {(acc)*100:.2f}% | Tiempo: {elapsed:.2f} s")
    

    
    # animación   
    print(f"Animando la trayectoria...")
    
    L_init = L_frames[0] 
    
    fig, ax = plt.subplots(figsize=(6, 8) if TOPOLOGY == 1 else (6, 6))
    ax.set_aspect('equal')
                           
    if TOPOLOGY == 0:
        ax.set_xlim(0, L_init)
        ax.set_ylim(0, L_init)
        ax.set_xticks(np.arange(0, L_init + 1, L_init))
        ax.set_yticks(np.arange(0, L_init + 1, L_init))
        boundary_coords = [(0,0), (L_init,0), (L_init,L_init), (0,L_init)]
        
    if TOPOLOGY == 1:
        ax.set_xlim(0, 3*L_init)
        ax.set_ylim(0, 4*L_init)
        ax.set_xticks(np.arange(0, 3*L_init + 1, L_init))
        ax.set_yticks(np.arange(0, 4*L_init + 1, L_init))
        boundary_coords = [(L_init,0), (2*L_init,0), (2*L_init,L_init), (3*L_init,L_init), 
                           (3*L_init,2*L_init), (2*L_init,2*L_init), (2*L_init,4*L_init), 
                           (L_init,4*L_init), (L_init,2*L_init), (0,2*L_init), (0,L_init), (L_init,L_init)]     
        
    ax.grid(True, linestyle='--', color='gray', alpha=0.5, zorder=0)
    ax.axhline(0, color='gray', linewidth=1.0, zorder=1)
    ax.axvline(0, color='gray', linewidth=1.0, zorder=1)                 
                           
    boundary_patch = Polygon(boundary_coords, closed=True, fill=False, edgecolor='black', lw=0.5, zorder=2)
    ax.add_patch(boundary_patch)
    
    MAX_CIRCLES = int(N * 4) 
    circles = [Circle((-100, -100), R, color='crimson', alpha=1.0, zorder=3) for _ in range(MAX_CIRCLES)]
    
    for c in circles:
        ax.add_patch(c)
        c.set_clip_path(boundary_patch)

    def init():
        return [boundary_patch] + circles

    def update(frame):
        frame_x = trayectoria[frame, :, 0]
        frame_y = trayectoria[frame, :, 1]
        L_actual = L_frames[frame] 
        
        if TOPOLOGY == 0:
            ax.set_xlim(0, L_actual)
            ax.set_ylim(0, L_actual)
            ax.set_xticks(np.arange(0, L_actual + 1e-5, L_actual))
            ax.set_yticks(np.arange(0, L_actual + 1e-5, L_actual))
            new_coords = [(0,0), (L_actual,0), (L_actual,L_actual), (0,L_actual)]
        elif TOPOLOGY == 1:
            ax.set_xlim(0, 3*L_actual)
            ax.set_ylim(0, 4*L_actual)
            ax.set_xticks(np.arange(0, 3*L_actual + 1e-5, L_actual))
            ax.set_yticks(np.arange(0, 4*L_actual + 1e-5, L_actual))
            new_coords = [(L_actual,0), (2*L_actual,0), (2*L_actual,L_actual), (3*L_actual,L_actual), 
                          (3*L_actual,2*L_actual), (2*L_actual,2*L_actual), (2*L_actual,4*L_actual), 
                          (L_actual,4*L_actual), (L_actual,2*L_actual), (0,2*L_actual), (0,L_actual), (L_actual,L_actual)]     
        
        boundary_patch.set_xy(new_coords)
        
        circle_idx = 0
        for i in range(N):
            if circle_idx < MAX_CIRCLES:
                circles[circle_idx].center = (frame_x[i], frame_y[i])
                circle_idx += 1
                
            ghosts = get_ghosts(frame_x[i], frame_y[i], L_actual, R, TOPOLOGY) 
            for j in range(8):
                gx, gy = ghosts[j]
                if gx != -100.0 and circle_idx < MAX_CIRCLES:
                    circles[circle_idx].center = (gx, gy)
                    circle_idx += 1
                    
        while circle_idx < MAX_CIRCLES:
            circles[circle_idx].center = (-100, -100)
            circle_idx += 1
            
        return [boundary_patch] + circles

    # tengo que acordarme que el blit=False es para que la escala de los ejes se actualice visualmente
    ani = animation.FuncAnimation(fig, update, frames=NUM_FRAMES, init_func=init, blit=False, interval=30)
    plt.show()
    
    
    
    
    
    
    
   
                           
    

    