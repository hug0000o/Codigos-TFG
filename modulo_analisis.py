import numpy as np
from numba import njit
import modulo_motor as mm


def get_delta_phi(phi_val, dphi_low=0.03, dphi_trans=0.015, dphi_high=0.01):
    if phi_val < 0.60:
        return dphi_low
    elif phi_val < 0.78:
        return dphi_trans
    else:
        return dphi_high


@njit(fastmath=True, boundscheck=False, cache=True)
def acumular_histograma_gr(x, y, L, dr, hist, topology):
    N_total = len(x)
    n_bins = len(hist)
    r_max = n_bins * dr

    for i in range(N_total - 1):
        xi, yi = x[i], y[i]
        for j in range(i + 1, N_total):
            r = mm.dist_g(xi, yi, x[j], y[j], L, topology)
            if r < r_max:
                ind = int(r / dr)
                if ind < n_bins:
                    hist[ind] += 2


def normalizar_gr(hist, n_cnf, N_part, rho, dr):
    n_bins = len(hist)
    edges = np.arange(n_bins + 1, dtype=np.float64) * dr
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    area_anillos = np.pi * (edges[1:]**2 - edges[:-1]**2)

    g = hist / (n_cnf * N_part * rho * area_anillos)
    return r_centers, g


def extraer_g_contacto(r, g, D_val, n_puntos=5):
    mask = (r > D_val) & (g > 0.0)
    r_valid = r[mask][:n_puntos]
    g_valid = g[mask][:n_puntos]

    if len(r_valid) < 2:
        return g_valid[0] if len(g_valid) > 0 else 0.0

    m, b = np.polyfit(r_valid, np.log(g_valid), 1)
    g_contact = np.exp(m * D_val + b)
    return g_contact


def calcular_presion(phi, g_contacto):
    Z = 1.0 + 2.0 * phi * g_contacto
    P_red = phi * Z
    return Z, P_red


def medir_estado(x_pos, y_pos, L, D_eff, max_disp, head, next_node, ncx, ncy, topology,
                 D_val, area_particulas, dr=0.01, phi_slow_start=0.60,
                 n_therm_low=300, n_cnf_low=300, n_therm_high=5000, n_cnf_high=1000,
                 sample_every=5, n_puntos_fit=5):
   
    area_box = mm.get_area(L, topology)
    phi = area_particulas / area_box
    rho = len(x_pos) / area_box

    if phi < phi_slow_start:
        n_therm = n_therm_low
        n_cnf = n_cnf_low
    else:
        n_therm = n_therm_high
        n_cnf = n_cnf_high

    for _ in range(n_therm):
        mm.mc_step(x_pos, y_pos, L, D_eff, max_disp, head, next_node, ncx, ncy, topology)

    r_max = 0.5 * L
    n_bins = int(r_max / dr)
    hist = np.zeros(n_bins, dtype=np.int64)

    for _ in range(n_cnf):
        for _ in range(sample_every):
            mm.mc_step(x_pos, y_pos, L, D_eff, max_disp, head, next_node, ncx, ncy, topology)
        acumular_histograma_gr(x_pos, y_pos, L, dr, hist, topology)

    r, g = normalizar_gr(hist, n_cnf, len(x_pos), rho, dr)
    g_c = extraer_g_contacto(r, g, D_val, n_puntos_fit)
    Z, P_red = calcular_presion(phi, g_c)

    return phi, Z, P_red, g_c, r, g