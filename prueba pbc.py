import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import matplotlib.animation as animation
from itertools import product, combinations

"""Script donde he diseñado y probado si funcionan correctamente las condiciones de contorno periódicas (PBC) 
para la cruz 2D. Como método extra de comprobación, se ha implementado la proyección 3D del sistema para
visualizar mejor su comportamiento. """



# Parámetros.
L = 10.0
D = 3.0                 
R = D / 2.0
DT = 0.04
STEPS = 1000


# Posición y velocidad inicial del disco.
current_x, current_y = 1.5 * L, 0.5 * L
current_vx, current_vy = 8.0, 11.0


# Nube de puntos para que el disco se doble bien en la proyección 3D.
res = R / 25.0 
dx_list, dy_list = [], []
for i in np.arange(-R, R, res):
    for j in np.arange(-R, R, res):
        if i**2 + j**2 <= R**2:
            dx_list.append(i)
            dy_list.append(j)
puntos_dx = np.array(dx_list)
puntos_dy = np.array(dy_list)



# Cinemática y topología. 
def inside_cross(x, y, L):
    if L <= x <= 2*L and 0 <= y <= 4*L: return True
    if 0 <= x <= 3*L and L <= y <= 2*L: return True
    return False



"""Función PBC. Esta versión incluye saltos en la velocidad, pero al pasar esta función al script principal hay que modificarla 
   para solo hacer cambios en la posición. Está así porque para comprobar que las PBC funcionan correctamente, he añadido el 
   disco que se mueve para que así vaya cruzando todas las aristas y comprobar que funcionan correctamente."""

def pbc(old_x, old_y, new_x, new_y, vx, vy, L):
    if inside_cross(new_x, new_y, L): return new_x, new_y, vx, vy
        
    if old_y < L: # Cara 1
        if new_y < 0: return new_x, new_y + 4*L, vx, vy
        if new_x < L: return new_y, 2*L - new_x, vy, -vx
        if new_x > 2*L: return 3*L - new_y, new_x - L, -vy, vx
    elif old_y > 3*L: # Cara 4
        if new_y > 4*L: return new_x, new_y - 4*L, vx, vy
        if new_x < L: return L - new_x, 5*L - new_y, -vx, -vy
        if new_x > 2*L: return 5*L - new_x, 5*L - new_y, -vx, -vy
    elif old_x < L: # Cara 5
        if new_y < L: return 2*L - new_y, new_x, -vy, vx
        if new_x < 0: return L - new_x, 5*L - new_y, -vx, -vy
        if new_y > 2*L: return new_y - L, 3*L - new_x, vy, -vx
    elif old_x > 2*L: # Cara 6
        if new_y < L: return L + new_y, 3*L - new_x, vy, -vx
        if new_x > 3*L: return 5*L - new_x, 5*L - new_y, -vx, -vy
        if new_y > 2*L: return 4*L - new_y, new_x, -vy, vx
    else: # Caras 2 y 3
        if 2*L <= old_y <= 3*L: 
            if new_x < L: return 3*L - new_y, L + new_x, -vy, vx
            if new_x > 2*L: return new_y, 4*L - new_x, vy, -vx
            
    return new_x, new_y, vx, vy

"""Las 14 condiciones topológicas necesarias para generar proyecciones del disco en aristas."""

def ghost_positions(x, y, L, R):
    ghosts = []
    if y < R and L <= x <= 2*L: ghosts.append((x, y + 4*L))
    if y > 4*L - R and L <= x <= 2*L: ghosts.append((x, y - 4*L))
    if x < L + R and 0 <= y <= L: ghosts.append((y, 2*L - x))
    if y < L + R and 0 <= x <= L: ghosts.append((2*L - y, x))
    if x > 2*L - R and 0 <= y <= L: ghosts.append((3*L - y, x - L))
    if y < L + R and 2*L <= x <= 3*L: ghosts.append((L + y, 3*L - x))
    if x < L + R and 2*L <= y <= 3*L: ghosts.append((3*L - y, L + x))
    if y > 2*L - R and 0 <= x <= L: ghosts.append((y - L, 3*L - x))
    if x > 2*L - R and 2*L <= y <= 3*L: ghosts.append((y, 4*L - x))
    if y > 2*L - R and 2*L <= x <= 3*L: ghosts.append((4*L - y, x))
    if x < L + R and 3*L <= y <= 4*L: ghosts.append((L - x, 5*L - y))
    if x < R and L <= y <= 2*L: ghosts.append((L - x, 5*L - y))
    if x > 2*L - R and 3*L <= y <= 4*L: ghosts.append((5*L - x, 5*L - y))
    if x > 3*L - R and L <= y <= 2*L: ghosts.append((5*L - x, 5*L - y))
    return ghosts

"""Filtro para recortar los discos fantasmas fuera de la cruz."""

def inside_cross_vec(px, py, L):
    m1 = (L <= px) & (px <= 2*L) & (0 <= py) & (py <= 4*L)
    m2 = (0 <= px) & (px <= 3*L) & (L <= py) & (py <= 2*L)
    return m1 | m2


# Animaciones.
# A partir de aquí tengo que revisar bien todo porque lo he hecho directamente con Gemini (Parece estar bien y funcionar sin problemas).

"""Proyección de las 6 caras planas de la cruz a los planos del cubo 3D."""

def map_2d_to_3d_vectorized(px, py, L):
    X3, Y3, Z3 = np.zeros_like(px), np.zeros_like(py), np.zeros_like(px)
    
    m = (L <= px) & (px <= 2*L) & (0 <= py) & (py <= L)
    X3[m], Y3[m], Z3[m] = px[m] - L, py[m], 0.0
    
    m = (L <= px) & (px <= 2*L) & (L < py) & (py <= 2*L)
    X3[m], Y3[m], Z3[m] = px[m] - L, L, py[m] - L
    
    m = (L <= px) & (px <= 2*L) & (2*L < py) & (py <= 3*L)
    X3[m], Y3[m], Z3[m] = px[m] - L, 3*L - py[m], L
    
    m = (L <= px) & (px <= 2*L) & (3*L < py) & (py <= 4*L)
    X3[m], Y3[m], Z3[m] = px[m] - L, 0.0, 4*L - py[m]
    
    m = (0 <= px) & (px < L) & (L < py) & (py <= 2*L)
    X3[m], Y3[m], Z3[m] = 0.0, px[m], py[m] - L
    
    m = (2*L < px) & (px <= 3*L) & (L < py) & (py <= 2*L)
    X3[m], Y3[m], Z3[m] = L, 3*L - px[m], py[m] - L
    
    return X3, Y3, Z3


fig = plt.figure(figsize=(14, 6))
caras_info = [
    (1.5*L, 0.5*L, '1'), (1.5*L, 1.5*L, '2'), (1.5*L, 2.5*L, '3'), 
    (1.5*L, 3.5*L, '4'), (0.5*L, 1.5*L, '5'), (2.5*L, 1.5*L, '6')
]


"""Parte 2D""" 

ax2d = fig.add_subplot(1, 2, 1)
ax2d.set_xlim(-L, 4*L)
ax2d.set_ylim(-L, 5*L)
ax2d.set_aspect('equal')
ax2d.set_title(f"Cubo Desdoblado 2D (L={L}, D={D})")
ax2d.set_xticks(np.arange(-L, 4*L + 1, L))
ax2d.set_yticks(np.arange(-L, 5*L + 1, L))
ax2d.grid(True, linestyle='--', color='gray', alpha=0.5, zorder=0)

cross_coords = [(L,0), (2*L,0), (2*L,L), (3*L,L), (3*L,2*L), (2*L,2*L), 
                (2*L,4*L), (L,4*L), (L,2*L), (0,2*L), (0,L), (L,L)]
cross = Polygon(cross_coords, closed=True, fill=False, edgecolor='black', lw=1.5, zorder=2)
ax2d.add_patch(cross)

for cx, cy, num in caras_info:
    ax2d.text(cx, cy, num, fontsize=40, ha='center', va='center', color='black', alpha=0.15, fontweight='bold', zorder=1)

MAX_CIRCLES = 5
circles = [Circle((-100, -100), R, color='crimson', alpha=1.0, zorder=3) for _ in range(MAX_CIRCLES)]
for c in circles:
    ax2d.add_patch(c)
    c.set_clip_path(cross)

"""Parte 3D"""
ax3d = fig.add_subplot(1, 2, 2, projection='3d')
ax3d.set_box_aspect([1,1,1])
ax3d.set_title(f"Cubo 3D (L={L}, D={D})")
ax3d.axis('off')
ax3d.view_init(elev=30, azim=-45)

vertices = [0, L]
for s, e in combinations(np.array(list(product(vertices, vertices, vertices))), 2):
    if np.sum(np.abs(s-e)) == L:
        ax3d.plot3D(*zip(s, e), color="black", lw=1.5, alpha=0.5)

def center_map_2d_to_3d(x, y, L):
    if L <= x <= 2*L and 0 <= y <= L: return x - L, y, 0.0               
    elif L <= x <= 2*L and L < y <= 2*L: return x - L, L, y - L             
    elif L <= x <= 2*L and 2*L < y <= 3*L: return x - L, 3*L - y, L           
    elif L <= x <= 2*L and 3*L < y <= 4*L: return x - L, 0.0, 4*L - y         
    elif 0 <= x < L and L < y <= 2*L: return 0.0, x, y - L               
    elif 2*L < x <= 3*L and L < y <= 2*L: return L, 3*L - x, y - L           
    return 0.0, 0.0, 0.0 

for cx, cy, num in caras_info:
    X, Y, Z = center_map_2d_to_3d(cx, cy, L)
    ax3d.text(X, Y, Z, num, fontsize=40, ha='center', va='center', color='black', alpha=0.2, fontweight='bold')

disco_3d, = ax3d.plot([], [], [], '.', color='crimson', markersize=4, alpha=1.0)


"""Bucle de animación"""

def update(frame):
    global current_x, current_y, current_vx, current_vy
    
    new_x = current_x + current_vx * DT
    new_y = current_y + current_vy * DT
    current_x, current_y, current_vx, current_vy = pbc(
        current_x, current_y, new_x, new_y, current_vx, current_vy, L
    )
    
    # 2D Nativo
    for c in circles: c.center = (-100, -100)
    circles[0].center = (current_x, current_y)
    
    ghosts = ghost_positions(current_x, current_y, L, R)
    for i, (gx, gy) in enumerate(ghosts):
        if i + 1 < MAX_CIRCLES: circles[i+1].center = (gx, gy)
            
    # Mapeo 3D Robusto (Basado en nubes locales recortadas por la cruz 2D)
    X3_total, Y3_total, Z3_total = [], [], []
    centers_to_draw = [(current_x, current_y)] + ghosts
    
    for cx, cy in centers_to_draw:
        px = cx + puntos_dx
        py = cy + puntos_dy
        
        # Filtramos estrictamente los puntos que caen en el plano de la cruz
        mask_inside = inside_cross_vec(px, py, L)
        px_valid = px[mask_inside]
        py_valid = py[mask_inside]
        
        # Proyectamos
        vx, vy, vz = map_2d_to_3d_vectorized(px_valid, py_valid, L)
        X3_total.extend(vx)
        Y3_total.extend(vy)
        Z3_total.extend(vz)
        
    disco_3d.set_data(X3_total, Y3_total)
    disco_3d.set_3d_properties(Z3_total)

ani = animation.FuncAnimation(fig, update, frames=STEPS, interval=20, blit=False)
plt.show()