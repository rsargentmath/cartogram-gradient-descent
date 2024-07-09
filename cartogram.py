import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple
import json
from time import perf_counter
from matplotlib.colors import rgb2hex


with open("ne_110m_admin_0_countries_lakes_FIXED.json", "r") as f:
    import_data = json.load(f)
WORLD_BORDERS_DATA = {}
WORLD_BORDERS_DATA_FLAT = {}
for k, v in import_data.items():
    nonflat_polygon_list = []
    flat_polygon_list = []
    for polygon_list in v:
        this_polygon_list = []
        for polygon in polygon_list:
            this_polygon_list.append(np.deg2rad(np.array(polygon)))
            flat_polygon_list.append(np.deg2rad(np.array(polygon)))
        nonflat_polygon_list.append(this_polygon_list)
    WORLD_BORDERS_DATA[k] = nonflat_polygon_list
    WORLD_BORDERS_DATA_FLAT[k] = flat_polygon_list
del import_data
POPULATION_DICT = {"Afghanistan": 38928340, "Albania": 2837850, "Algeria": 43851040, "Andorra": 77270, "Angola": 32866270, "Antigua and Barbuda": 97930,
			"Argentina": 45376760, "Armenia": 2963230, "Australia": 25693270, "Austria": 8916860, "Azerbaijan": 10093120, "The Bahamas": 393250,
			"Bahrain": 1701580, "Bangladesh": 164689380, "Barbados": 287370, "Belarus": 9379950, "Belgium": 11544240, "Belize": 397620, "Benin": 12123200,
			"Bhutan": 771610, "Bolivia": 11673030, "Bosnia and Herzegovina": 3280820, "Botswana": 2351630, "Brazil": 212559410,
			"Brunei": 437480, "Bulgaria": 6934020, "Burkina Faso": 20903280, "Burundi": 11890780, "Cabo Verde": 555990, "Cambodia": 16718970, "Cameroon": 26545860,
			"Canada": 38037200, "Central African Republic": 4829760, "Chad": 16425860, "Chile": 19116210,
			"China": 1410929360, "Colombia": 50882880, "Comoros": 869600, "Democratic Republic of the Congo": 89561400, "Republic of the Congo": 5518090, "Costa Rica": 5094110, "Ivory Coast": 26378280,
			"Croatia": 4047680, "Cuba": 11326620, "Cyprus": 881360, "Czechia": 10697860, "Denmark": 5831400, "Djibouti": 988000, "Dominica": 71990,
			"Dominican Republic": 10847900, "Ecuador": 17643060, "Egypt": 102334400, "El Salvador": 6486200, "Equatorial Guinea": 1402980, "Eritrea": 3213970,
			"Estonia": 1329480, "eSwatini": 1160160, "Ethiopia": 114963580, "Fiji": 896440, "Finland": 5529540, "France": 67379910,
			"Gabon": 2225730, "Gambia": 2416660, "Georgia": 3722720, "Germany": 83160870, "Ghana": 31072940,
			"Greece": 10700560, "Grenada": 112520, "Guatemala": 16858330, "Guinea": 13132790, "Guinea-Bissau": 1968000, "Guyana": 786560,
			"Haiti": 11402530, "Honduras": 9904610, "Hungary": 9750150, "Iceland": 366460, "India": 1380004390, "Indonesia": 273523620,
			"Iran": 83992950, "Iraq": 40222500, "Ireland": 4985670, "Israel": 14018370, "Italy": 59449530, "Jamaica": 2961160,
			"Japan": 125836020, "Jordan": 10203140, "Kazakhstan": 18754440, "Kenya": 53771300, "Kiribati": 119450, "North Korea": 25778810, "South Korea": 51836240,
			"Kosovo": 1790130, "Kuwait": 4270560, "Kyrgyzstan": 6579900, "Laos": 7275560, "Latvia": 1900450, "Lebanon": 6825440, "Lesotho": 2142250,
			"Liberia": 5057680, "Libya": 6871290, "Liechtenstein": 38140, "Lithuania": 2794890, "Luxembourg": 630420, "Madagascar": 27691020,
			"Malawi": 19129960, "Malaysia": 32366000, "Maldives": 540540, "Mali": 20250830, "Malta": 515330, "Marshall Islands": 59190, "Mauritania": 4649660,
			"Mauritius": 1265740, "Mexico": 128932750, "Federated States of Micronesia": 115020, "Moldova": 2620490, "Monaco": 39240, "Mongolia": 3278290, "Montenegro": 621310,
			"Morocco": 36343160, "Mozambique": 31255440, "Myanmar": 54409790, "Namibia": 2540920, "Nauru": 10830, "Nepal": 29136810, "Netherlands": 17441500,
			"New Zealand": 5084300, "Nicaragua": 6624550, "Niger": 24206640, "Nigeria": 206139590, "North Macedonia": 2072530,
			"Norway": 5379480, "Oman": 5106620, "Pakistan": 220892330, "Palau": 18090, "Panama": 4314770, "Papua New Guinea": 8947030,
			"Paraguay": 7132530, "Peru": 32971850, "Philippines": 109581090, "Poland": 37899070, "Portugal": 10297080, "Qatar": 2881060,
			"Romania": 19257520, "Russia": 144104080, "Rwanda": 12952210, "Samoa": 198410, "San Marino": 33940, "São Tomé and Principe": 219160,
			"Saudi Arabia": 34813870, "Senegal": 16743930, "Republic of Serbia": 6899130, "Seychelles": 98460, "Sierra Leone": 7976980, "Singapore": 5685810,
			"Slovakia": 5458830, "Slovenia": 2102420, "Solomon Islands": 686880, "Somalia": 10193220, "South Africa": 59308690,
			"South Sudan": 11193730, "Spain": 47363420, "Sri Lanka": 21919000, "Saint Kitts and Nevis": 53190, "Saint Lucia": 183630,
			"Saint Vincent and the Grenadines": 110950, "Sudan": 43849270, "Suriname": 586630, "Sweden": 10353440, "Switzerland": 8636560, "Syria": 17500660,
			"Tajikistan": 9537640, "United Republic of Tanzania": 59734210, "Thailand": 69799980, "East Timor": 1318440, "Togo": 8278740, "Tonga": 105700, "Trinidad and Tobago": 1399490,
			"Tunisia": 11818620, "Turkey": 84339070, "Turkmenistan": 6031190, "Tuvalu": 11790, "Uganda": 45741000, "Ukraine": 44134690,
			"United Arab Emirates": 9890400, "United Kingdom": 67215290, "United States of America": 331501080, "Uruguay": 3473730, "Uzbekistan": 34232050, "Vanuatu": 307150,
			"Venezuela": 28435940, "Vietnam": 97338580, "Yemen": 29825970, "Zambia": 18383960,
			"Zimbabwe": 14862930, "Antarctica": 3000, "Northern Cyprus": 326000, "Western Sahara": 567400, "Somaliland": 5700000, "Taiwan": 23451840, "Vatican": 800, "Kashmir": 1000}
POPULATION_ARRAY = np.array([POPULATION_DICT[k]
                             for k in WORLD_BORDERS_DATA_FLAT.keys()],
                            dtype=float)


TOLERANCE = 1e-12
PHI = (1 + np.sqrt(5)) / 2


class Mesh(NamedTuple):
    verts: np.ndarray
    tris: np.ndarray


class ValueGrad(NamedTuple):
    value: np.ndarray   # or could be just a float
    grad: np.ndarray


OCTAHEDRON = Mesh(verts=np.array([[1, 0, 0], [0, 1, 0],
                                  [0, 0, 1], [-1, 0, 0],
                                  [0, -1, 0], [0, 0, -1]]),
                  tris=np.array([[0, 1, 2], [0, 2, 4],
                                 [0, 4, 5], [0, 5, 1],
                                 [1, 3, 2], [1, 5, 3],
                                 [2, 3, 4], [3, 5, 4]])
                  )
ICOSAHEDRON = Mesh(verts=(np.array([[PHI, 1, 0], [-PHI, 1, 0],
                                   [-PHI, -1, 0], [PHI, -1, 0],
                                   [0, PHI, 1], [0, -PHI, 1],
                                   [0, -PHI, -1], [0, PHI, -1],
                                   [1, 0, PHI], [1, 0, -PHI],
                                   [-1, 0, -PHI], [-1, 0, PHI]])
                          / np.sqrt(PHI + 2)),
                   tris=np.array([[0, 3, 9], [0, 4, 8],
                                  [0, 7, 4], [0, 8, 3],
                                  [0, 9, 7], [1, 2, 11],
                                  [1, 4, 7], [1, 7, 10],
                                  [1, 10, 2], [1, 11, 4],
                                  [2, 5, 11], [2, 6, 5],
                                  [2, 10, 6], [3, 5, 6],
                                  [3, 6, 9], [3, 8, 5],
                                  [4, 11, 8], [5, 8, 11],
                                  [6, 10, 9], [7, 9, 10]])
                   )
a = 1 / np.sqrt(3)
CUBE = Mesh(verts=np.array([[1, 0, 0], [0, 1, 0],
                            [0, 0, 1], [-1, 0, 0],
                            [0, -1, 0], [0, 0, -1],
                            [a, a, a], [-a, a, a],
                            [a, -a, a], [-a, -a, a],
                            [a, a, -a], [-a, a, -a],
                            [a, -a, -a], [-a, -a, -a]]),
            tris=np.array([[0, 6, 8], [0, 8, 12],
                           [0, 10, 6], [0, 12, 10],
                           [1, 6, 10], [1, 7, 6],
                           [1, 10, 11], [1, 11, 7],
                           [2, 6, 7], [2, 7, 9],
                           [2, 8, 6], [2, 9, 8],
                           [3, 7, 11], [3, 9, 7],
                           [3, 11, 13], [3, 13, 9],
                           [4, 8, 9], [4, 9, 13],
                           [4, 12, 8], [4, 13, 12],
                           [5, 10, 12], [5, 11, 10],
                           [5, 12, 13], [5, 13, 11]])
            )
del a


def matrix_times_array_of_vectors(matrix, vectors):
    return (matrix @ vectors[..., np.newaxis])[..., 0]


def subdivide_tri(a, b, c, n):
    verts = [a]
    tris = []
    edge_left = np.linspace(a, b, n + 1)
    edge_right = np.linspace(a, c, n + 1)
    ix = 0
    for i in range(n):
        edge_horiz = np.linspace(edge_left[i + 1], edge_right[i + 1], i + 2)
        verts += list(edge_horiz)
        tris.append([ix, ix + i + 1, ix + i + 2])
        for j in range(i):
            tris.append([ix + j, ix + i + j + 2, ix + j + 1])
            tris.append([ix + j + 1, ix + i + j + 2, ix + i + j + 3])
        ix += i + 1
    return Mesh(verts=np.array(verts), tris=np.array(tris))


def tri_edge_vert_indices(n):
    if n == 0:
        return np.array([0])
    edge_ixs = [0]
    left_ix = 1
    for i in range(1, n):
        edge_ixs += [left_ix, left_ix + i]
        left_ix += i + 1
    edge_ixs += range(left_ix, left_ix + n + 1)
    return np.array(edge_ixs)


def vec_norm(v):
    return np.sqrt(v @ v)


def nzd(v):
    return v / vec_norm(v)


def dot_flat(v0, v1):
    return np.sum(v0 * v1)


def line_intersection_plane(a0, a1, b0, b1, infinite_a=False, infinite_b=False):
    assert a0.shape[0] == 2
    if abs(np.cross(a1-a0, b1-b0)) < TOLERANCE:     # if close to parallel
        return None
    norm_a = np.array([[0, -1], [1, 0]]) @ nzd(a1 - a0)
    norm_b = np.array([[0, -1], [1, 0]]) @ nzd(b1 - b0)
    dot_a = norm_a @ a0
    dot_b = norm_b @ b0
    dot_n = norm_a @ norm_b
    coeff_a = (dot_a - dot_n*dot_b) / (1 - dot_n*dot_n)
    coeff_b = (dot_b - dot_n*dot_a) / (1 - dot_n*dot_n)
    point = coeff_a * norm_a + coeff_b * norm_b
    min_cross_a, max_cross_a = sorted([np.cross(norm_a, a0),
                                       np.cross(norm_a, a1)])
    min_cross_b, max_cross_b = sorted([np.cross(norm_b, b0),
                                       np.cross(norm_b, b1)])
    cross_a = np.cross(norm_a, point)
    cross_b = np.cross(norm_b, point)
    if not infinite_a:
        if cross_a > max_cross_a or cross_a < min_cross_a:
            return None
    if not infinite_b:
        if cross_b > max_cross_b or cross_b < min_cross_b:
            return None
    return point


def gnomonic_projection(v):
    assert v.shape == (3,)
    assert v[2] > TOLERANCE
    return v[0:2] / v[2]


def portion_of_tri_inside_polygon_sphere(a, b, c, polygon):
    distance_threshold = 0.22   # roughly the radius of the biggest circle
                                # that can be inscribed in Russia
    def dist_to_a(vertex):
        return vec_norm(vertex - a)
    distances = np.apply_along_axis(dist_to_a, 1, polygon)
    if np.min(distances) > distance_threshold:
        return 0
    mat = tangent_space_matrix(a, b, c, clamp_to_sphere=True)
    def rotate_and_project(v):
        return gnomonic_projection(mat.T @ v)
    a_proj = rotate_and_project(a)
    b_proj = rotate_and_project(b)
    c_proj = rotate_and_project(c)
    polygon_proj = np.apply_along_axis(rotate_and_project, 1, polygon)
    return portion_of_tri_inside_polygon_plane(a_proj,
                                               b_proj,
                                               c_proj,
                                               polygon_proj)


def portion_of_tri_inside_polygon_plane(a, b, c, polygon):
    to_std_tri_mat = np.linalg.inv(np.column_stack([b-a, c-a]))
    def transform(v):
        return to_std_tri_mat @ (v - a)
    poly_trans = np.apply_along_axis(transform, 1, polygon)
    area_portion = 0
    for i in range(poly_trans.shape[0]):
        pt0 = poly_trans[i]
        pt1 = poly_trans[(i+1) % poly_trans.shape[0]]
        if pt1[0] <= pt0[0]:
            orientation = 1
            pt_left = pt1
            pt_right = pt0
        else:
            orientation = -1
            pt_left = pt0
            pt_right = pt1
        if pt_left[0] <= 0 and pt_right[0] <= 0:
            continue
        if pt_left[0] >= 1 and pt_right[0] >= 1:
            continue
        left_intersection = line_intersection_plane(pt_left,
                                                    pt_right,
                                                    np.array([0, 0]),
                                                    np.array([0, 1]),
                                                    infinite_b=True)
        right_intersection = line_intersection_plane(pt_left,
                                                     pt_right,
                                                     np.array([1, 0]),
                                                     np.array([1, 1]),
                                                     infinite_b=True)
        if left_intersection is not None:
            pt_left = left_intersection
        if right_intersection is not None:
            pt_right = right_intersection

        bottom_intersection = line_intersection_plane(pt_left,
                                                      pt_right,
                                                      np.array([0, 0]),
                                                      np.array([1, 0]),
                                                      infinite_b=False)
        top_intersection = line_intersection_plane(pt_left,
                                                   pt_right,
                                                   np.array([1, 0]),
                                                   np.array([0, 1]),
                                                   infinite_b=False)
        pts = [pt_left, pt_right]
        if bottom_intersection is not None:
            pts.append(bottom_intersection)
        if top_intersection is not None:
            pts.append(top_intersection)
        pts_sorted = np.array(sorted(pts, key=lambda pt: pt[0]))
        def clamp_to_tri(pt):
            pt_y_clamped = np.clip(pt[1], 0, 1 - pt[0])
            return np.array([pt[0], pt_y_clamped])
        pts_clamped = np.apply_along_axis(clamp_to_tri, 1, pts_sorted)
        for j in range(pts_clamped.shape[0] - 1):
            pt_l, pt_r = pts_clamped[j], pts_clamped[j + 1]
            trapezoid_area = (pt_r[0] - pt_l[0]) * (pt_l[1] + pt_r[1]) / 2
            area_portion += 2 * orientation * trapezoid_area
    return area_portion


def lonlat_to_cartes(lonlat):
    lon, lat = lonlat
    return np.array([np.cos(lon) * np.cos(lat),
                    np.sin(lon) * np.cos(lat),
                    np.sin(lat)])


def tri_area_plane(a, b, c):
    if a.shape[0] == 2:
        return 1/2 * np.cross(b-a, c-a)
    return 1/2 * vec_norm(np.cross(b-a, c-a))


def tri_area_sphere(a, b, c):
    return 2 * np.arctan(a @ np.cross(b, c) / (1 + a@b + b@c + c@a))

    
def tri_scale_corrections(a, b, c):
    r0 = nzd(np.cross(b-a, c-a)) @ a    # distance from abc plane to origin
    abc_avg = (a + b + c) / 3
    scale_avg = tri_area_plane(a, b, c) / tri_area_sphere(a, b, c)
    alpha = 1 / np.sqrt(scale_avg * r0)
    beta = vec_norm(abc_avg)**(3/2) * alpha
    return alpha, beta


def bary_coords_to_abc_plane(a, b, c, barys):
    return matrix_times_array_of_vectors(np.column_stack([a, b, c]), barys)
    

# Approximation. Works best when abc is equilateral.        
def subdivide_tri_sphere(a, b, c, n):
    bary_coords, tris = subdivide_tri(
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        n,
    )
    alpha, beta = tri_scale_corrections(a, b, c)
    a1 = alpha
    a2 = (3 + 6*alpha - beta - 8*alpha*beta) / (2*beta - 3)
    a3 = 3 * (-2 - alpha + beta + 2*alpha*beta) / (2*beta - 3)
    def f(x):
        return a1*x + a2*x*x + a3*x*x*x
    def f_vec(v):
        return f(v) / np.sum(f(v))
    bary_coords_new = np.apply_along_axis(f_vec, 1, bary_coords)
    verts = bary_coords_to_abc_plane(a, b, c, bary_coords_new)
    verts_nzd = np.apply_along_axis(nzd, 1, verts)
    return Mesh(verts=verts_nzd, tris=tris)


# Mesh MUST be symmetric when reflecting across edges, otherwise
# vertices on edges may not line up.
def subdivide_mesh_sphere(mesh, n):
    # Subdivide each mesh triangle. Store new vertices and triangles.
    verts_og, tris_og = mesh.verts, mesh.tris
    verts_subdiv_list = []
    tris_subdiv_list = []
    verts_per_og_tri = ((n + 1) * (n + 2)) // 2
    for i, tri in enumerate(tris_og):
        verts_new, tris_new = subdivide_tri_sphere(*verts_og[tri], n)
        verts_subdiv_list.append(verts_new)
        tris_subdiv_list.append(tris_new + i*verts_per_og_tri)
    verts = np.concatenate(verts_subdiv_list)
    tris = np.concatenate(tris_subdiv_list)

    edge_vert_ixs = tri_edge_vert_indices(n)
    ixs_to_check = np.concatenate( # edge indices to check for duplicates
        [edge_vert_ixs + i*verts_per_og_tri for i in range(tris_og.shape[0])])
    first_seen_ixs = []
    old_ixs_to_new = np.arange(verts.shape[0])
    for ix in ixs_to_check:
        for seen_ix in first_seen_ixs:
            if vec_norm(verts[ix] - verts[seen_ix]) < TOLERANCE:
                old_ixs_to_new[ix] = seen_ix
                break
        else: # nobreak
            first_seen_ixs.append(ix)

    ixs_unique = np.unique(old_ixs_to_new)
    verts_final = verts[ixs_unique]
    new_ixs_to_final = np.arange(verts.shape[0])
    for i, val in enumerate(ixs_unique):
        new_ixs_to_final[val] = i
    old_ixs_to_final = new_ixs_to_final[old_ixs_to_new]
    tris_final = old_ixs_to_final[tris]
    return Mesh(verts=verts_final, tris=tris_final)


def tangent_space_matrix(a, b, c, clamp_to_sphere=False):
    dim = a.shape[0]
    assert dim == 2 or dim == 3
    if dim == 2:
        return np.identity(2)
    
    assert vec_norm(np.cross(b-a, c-a)) >= TOLERANCE
    t = nzd(b - a)
    n = nzd(np.cross(b-a, c-a))
    if clamp_to_sphere and n @ a < 0:
        n = -n
    s = np.cross(n, t)
    return np.column_stack([t, s, n])


# returns matrix that takes the vecs [1,0], [0,1] to b-a, c-a in tangent space
def matrix_basis_vecs_to_tri(a, b, c, clamp_to_sphere=False):
    dim = a.shape[0]
    assert dim == 2 or dim == 3
    if dim == 2:
        return np.column_stack([b-a, c-a])

    tan_space_mat = tangent_space_matrix(a, b, c, clamp_to_sphere)
    tan_space_inv = tan_space_mat.T
    return tan_space_inv[0:2] @ np.column_stack([b-a, c-a])


def gradient_descent(
        cost_grad_func, # returns ValueGrad
        initial_state,
        *,
        normalize_func=lambda x: x,
        iteration_count=100,
        memory=5,
        grad_tolerance=1e-5,
        ):
    x = initial_state.copy()
    s = []
    y = []
    rho = []
    H_0_scale = 1
    def H(k, g):    # approximation of the inverse hessian times g
        if k == 0:
            return H_0_scale * g    # H_0 is multiple of identity
        yps_g = g - rho[k-1] * y[k-1] * dot_flat(s[k-1], g)
        H_prev_yps_g = H(k-1, yps_g)
        return (H_prev_yps_g
                - rho[k-1] * s[k-1] * dot_flat(y[k-1], H_prev_yps_g)
                + rho[k-1] * s[k-1] * dot_flat(s[k-1], g))

    learn_rate = 1
    cost_grad = cost_grad_func(x)
    for i in range(iteration_count):
        g = cost_grad.grad
        if (np.abs(g) < grad_tolerance).all():
            return x
        search_dir = -H(len(s), g)
        x_new, cost_grad_new, learn_rate = line_search(cost_grad_func,
                                           x,
                                           cost_grad,
                                           search_dir,
                                           learn_rate,
                                           normalize_func=normalize_func)
        g_new = cost_grad_new.grad
        
        s.append(x_new - x)
        y.append(g_new - g)
        rho.append(1 / dot_flat(y[-1], s[-1]))
        s = s[-memory:]     # throw out oldest memory if necessary
        y = y[-memory:]
        rho = rho[-memory:]
        s0_dot_y0 = dot_flat(s[0], y[0])
        y0_dot_y0 = dot_flat(y[0], y[0])
        if s0_dot_y0 > TOLERANCE and y0_dot_y0 > TOLERANCE:
            H_0_scale = s0_dot_y0 / y0_dot_y0
        
        x = x_new
        cost_grad = cost_grad_new
    return x


def line_search(cost_grad_func,
                state,
                initial_cost_grad,
                search_dir,
                initial_learn_rate,
                *,
                normalize_func=lambda x: x,
                tau=0.5,
                c=0.1):
    print("START LINE SEARCH")
    m = dot_flat(search_dir, initial_cost_grad.grad)
    learn_rate = initial_learn_rate
    state_new = normalize_func(state + learn_rate * search_dir)
    cost_grad_new = cost_grad_func(state_new)
    if (cost_grad_new.value - initial_cost_grad.value
            <= c * learn_rate * m + TOLERANCE): # starting step is small enough
        print(f"{learn_rate} is small enough")
        for i in range(5):
            learn_rate_alt = learn_rate / tau   # try making the step bigger
            if learn_rate_alt > 1:      # step is now too big
                return state_new, cost_grad_new, learn_rate
            state_alt = normalize_func(state + learn_rate_alt * search_dir)
            cost_grad_alt = cost_grad_func(state_alt)
            if (cost_grad_alt.value - initial_cost_grad.value
                    > c * learn_rate_alt * m + TOLERANCE): # step is now too big
                print(f"{learn_rate_alt} is too big")
                return state_new, cost_grad_new, learn_rate
            # step is still small enough
            learn_rate = learn_rate_alt
            state_new = state_alt
            cost_grad_new = cost_grad_alt
            if i == 4:
                return state_new, cost_grad_new, learn_rate
    # starting step is too big
    print(f"{learn_rate} is too big")
    while True:
        learn_rate *= tau   # make the step smaller
        state_new = normalize_func(state + learn_rate * search_dir)
        cost_grad_new = cost_grad_func(state_new)
        if (cost_grad_new.value - initial_cost_grad.value
                <= c * learn_rate * m + TOLERANCE):  # step is now small enough
            print(f"{learn_rate} is small enough")
            return state_new, cost_grad_new, learn_rate


def clamp_inside_half_space(v, clamp_vec, min_dot):
    # clamp_vec has length 1
    if v @ clamp_vec >= min_dot:
        return v
    return v + clamp_vec * (min_dot - v @ clamp_vec)


def clamp_inside_tri_BAD(v, clamp_vecs, min_dots):
    # only works if v is outside at most one of the half spaces
    v_out = v.copy()
    for i in range(3):
        v_out = clamp_inside_half_space(v_out, clamp_vecs[i], min_dots[i])
    return v_out


def is_point_inside_tri(v, a, b, c):
    # If dimension is 3, checks if v is inside the infinite pyramid with
    # the vertex at the origin and edges through a, b, and c.
    if v.shape[0] == 2:
        return (np.cross(b-a, v-a) >= -TOLERANCE
                and np.cross(c-b, v-b) >= -TOLERANCE
                and np.cross(a-c, v-c) >= -TOLERANCE)
    return (np.cross(a, b) @ v >= -TOLERANCE
            and np.cross(b, c) @ v >= -TOLERANCE
            and np.cross(c, a) @ v >= -TOLERANCE)


def point_to_tri_plane(v, a, b, c):
    if v.shape[0] == 2:
        return v
    normal = nzd(np.cross(b-a, c-a))
    return v * (a @ normal) / (v @ normal)


def point_old_tri_to_new(v, a0, b0, c0, a1, b1, c1):
    assert v.shape[0] == a0.shape[0]
    if v.shape[0] == 2:
        mat0 = np.column_stack([b0-a0, c0-a0])
    elif v.shape[0] == 3:
        normal = nzd(np.cross(b0-a0, c0-a0))
        mat0 = np.column_stack([b0-a0, c0-a0, normal])
    mat1 = np.column_stack([b1-a1, c1-a1])
    mat = mat1 @ np.linalg.inv(mat0)[0:2]
    return a1 + mat @ (v - a0)


def point_old_mesh_to_new(v, verts_old, verts_new, tris):
    for tri in tris:
        if is_point_inside_tri(v, *verts_old[tri]):
            v_plane = point_to_tri_plane(v, *verts_old[tri])
            return point_old_tri_to_new(v, *verts_old[tri], *verts_new[tri])


def tri_det_frob_value_grads(a, b, c, G0, G):
    G0inv = np.linalg.inv(G0)
    # gradient where we're moving a, b, c using the tan space coords
    G_grad = np.array([[[[-1, -1], [0, 0]], [[0, 0], [-1, -1]]],
                       [[[1, 0], [0, 0]], [[0, 0], [1, 0]]],
                       [[[0, 1], [0, 0]], [[0, 0], [0, 1]]]])
    E = G @ G0inv
    E_grad = G_grad @ G0inv
    D = np.linalg.det(E)
    F = np.sum(E * E)
    D_grad = (E_grad[..., 0, 0] * E[1, 1]
              + E[0, 0] * E_grad[..., 1, 1]
              - E_grad[..., 0, 1] * E[1, 0]
              - E[0, 1] * E_grad[..., 1, 0])
    F_grad = 2 * (E[0, 0] * E_grad[..., 0, 0]
                  + E[0, 1] * E_grad[..., 0, 1]
                  + E[1, 0] * E_grad[..., 1, 0]
                  + E[1, 1] * E_grad[..., 1, 1])
    det_value_grad = ValueGrad(value=D, grad=D_grad)
    frob_value_grad = ValueGrad(value=F, grad=F_grad)
    return det_value_grad, frob_value_grad


def area_portions_array(mesh, borders_data_flat):
    
    def tri_area_portions(tri):
        portions = []
        for region in borders_data_flat.values():
            this_region_portion = 0
            for polygon_lonlat in region:
                polygon = np.apply_along_axis(lonlat_to_cartes,
                                              1,
                                              polygon_lonlat)
                this_region_portion += portion_of_tri_inside_polygon_sphere(
                        *mesh.verts[tri], polygon)
            portions.append(this_region_portion)
        return np.array(portions)
    
    return np.apply_along_axis(tri_area_portions, 1, mesh.tris)


def cartogram(mesh, portions, pop_array, max_iterations, clamp_to_sphere):
    verts_og, tris = mesh.verts, mesh.tris
    G0_array = np.apply_along_axis(
        lambda tri: matrix_basis_vecs_to_tri(*verts_og[tri], clamp_to_sphere),
        1,
        tris)
    M0_array = np.array([1/2 * np.linalg.det(G0) for G0 in G0_array])
    tri_region_areas_og = M0_array[:, np.newaxis] * portions
            # absolute area of each region falling in each tri
    region_areas_og = np.sum(tri_region_areas_og, axis=0)
    world_pop_density = np.sum(pop_array) / np.sum(region_areas_og)
    region_areas_intended = pop_array / world_pop_density
    region_scales_intended = region_areas_intended / region_areas_og
    land_portions = np.sum(portions, axis=1)
    tri_scales_intended = (np.sum(portions * region_scales_intended, axis=1)
                           + 1 - land_portions)
    print(np.min(region_scales_intended), np.max(region_scales_intended))
    print(list(WORLD_BORDERS_DATA_FLAT.keys())[np.argmin(region_scales_intended)],
          list(WORLD_BORDERS_DATA_FLAT.keys())[np.argmax(region_scales_intended)])

    def cost_grad_func_maker(weights_dist, weights_area, weight_error):

        def cost_grad_func(verts):
            cost = 0
            cost_grad = np.zeros_like(verts)
            G_array = np.apply_along_axis(
                lambda tri: matrix_basis_vecs_to_tri(*verts[tri],
                                                     clamp_to_sphere),
                1,
                tris)
            M_array = np.array([1/2 * np.linalg.det(G) for G in G_array])
            tolerance = 1e-14
            if np.min(M_array) < tolerance: # topology is violated
                return ValueGrad(value=np.inf, grad=np.zeros_like(verts))
            tri_region_areas = M_array[:, np.newaxis] * portions
            region_areas = np.sum(tri_region_areas, axis=0)
            region_errors = region_areas - region_areas_intended
            cost_error = np.sum(region_errors * region_errors)
            cost += weight_error * cost_error
            for i, tri in enumerate(tris):
                G0 = G0_array[i]
                M0 = M0_array[i]
                G = G_array[i]
                A = tri_scales_intended[i]
                tan_space_mat = tangent_space_matrix(*verts[tri],
                                                     clamp_to_sphere)
                (D, D_grad), (F, F_grad) = tri_det_frob_value_grads(*verts[tri],
                                                                    G0, G)
                cost_dist = M0 * (F/D - 2)
                cost_dist_grad = M0 * (F_grad*D - F*D_grad) / (D*D)
                cost_area = M0 * (D/A + A/D - 2)
                cost_area_grad = M0 * (D_grad / A - A*D_grad / (D*D))
                cost_error_grad = (2 * M0
                                   * np.sum(portions[i] * region_errors)
                                   * D_grad)

                cost += (weights_dist[i] * cost_dist
                         + weights_area[i] * cost_area)
                
                tri_cost_grad = (weights_dist[i] * cost_dist_grad
                                 + weights_area[i] * cost_area_grad
                                 + weight_error * cost_error_grad)
                tri_cost_grad_global_coords = matrix_times_array_of_vectors(
                                                       tan_space_mat[:, 0:2],
                                                       tri_cost_grad)
                for j in range(3):
                    cost_grad[tri[j]] += tri_cost_grad_global_coords[j]
            print(cost)
            return ValueGrad(cost, cost_grad)

        return cost_grad_func

    def normalize_func(verts):
        return np.apply_along_axis(nzd, 1, verts)

    verts_new = matrix_times_array_of_vectors(np.diag((1,1,1)), verts_og)
    
    weights_dist = np.full_like(M0_array, 0.05)
    weights_area = np.full_like(M0_array, 0.0)
    weight_error = 1
    verts_new = gradient_descent(cost_grad_func_maker(weights_dist,
                                                      weights_area,
                                                      weight_error),
                                 verts_new,
                                 iteration_count=max_iterations,
                                 normalize_func=normalize_func,
                                 grad_tolerance=1e-2)
    set_up_plot(1.02, 0)
    for i, tri in enumerate(mesh.tris):
        a, b, c = verts_new[tri]
        if a.shape[0] == 3:
            if np.cross(b-a, c-a)[2] <= 0:
                continue
        a2d, b2d, c2d = a[0:2], b[0:2], c[0:2]
        xs, ys = np.column_stack([a2d, b2d, c2d, a2d])
        color = rgb2hex((0, np.clip(land_portions[i], 0, 1), 0))
        plt.fill(xs, ys, color)
    return Mesh(verts=verts_new, tris=tris)


def octahedron_equal_area(it_count):
    mesh = subdivide_tri_sphere(*OCTAHEDRON.verts[OCTAHEDRON.tris[0]], 48)
    verts_og, tris = mesh.verts, mesh.tris
    num_verts = verts_og.shape[0]
    num_tris = tris.shape[0]
    G0_array = np.empty((num_tris, 2, 2))
    for i, tri in enumerate(tris):
        a, b, c = verts_og[tri]
        G0 = matrix_basis_vecs_to_tri(a, b, c)
        G0_array[i] = G0

    def cost_grad_func_maker(weight_dist):
        return lambda verts_state: cost_grad_func(verts_state, weight_dist)
    
    def cost_grad_func(verts_state, weight_dist=0.05):
        weight_area = 1
        #weight_dist = 0.05
        cost = 0
        grad_cost = np.zeros_like(verts_state)
        max_ratio_seen = 0
        for i, tri in enumerate(tris):
            a, b, c = verts_state[tri]
            A = 1   # desired area scale
            G0 = G0_array[i]
            M0 = 1/2 * np.linalg.det(G0)
            tan_space_mat = tangent_space_matrix(a, b, c, False)
            G = matrix_basis_vecs_to_tri(a, b, c, False)
            (D, D_grad), (F, F_grad) = tri_det_frob_value_grads(a, b, c, G0, G)
            this_tri_cost = M0 * (weight_dist * (F/D - 2)
                                  + weight_area * (D/A + A/D - 2))
            this_tri_cost_grad = M0 * (weight_dist * (F_grad*D - F*D_grad)/(D*D)
                                + weight_area * (D_grad/A - A*D_grad/(D*D)))
            this_tri_cost_grad_global_coords = matrix_times_array_of_vectors(
                                                   tan_space_mat[:, 0:2],
                                                   this_tri_cost_grad)
            cost += this_tri_cost
            for j in range(3):
                grad_cost[tri[j]] += this_tri_cost_grad_global_coords[j]
            max_ratio_seen = max(max_ratio_seen, D/A, A/D)
        print(max_ratio_seen, cost)
        return ValueGrad(value=cost, grad=grad_cost)

    def normalize_func(verts_state):
        b_over_a = 3**1.5 / 2 * 0.25681278
        a = np.sqrt( np.pi / (2 * (3**1.5 + (1 - 3**0.5) * b_over_a**2)) )

        def this_clamp_inside_tri(v):
            return clamp_inside_tri_BAD(v,
                                        np.array([[0, 1],
                                                  [-3**0.5/2, -1/2],
                                                  [3**0.5/2, -1/2]]),
                                        np.array([-a, -a, -a]))

        return np.apply_along_axis(this_clamp_inside_tri, 1, verts_state)

    rot_mat = tangent_space_matrix(*OCTAHEDRON.verts[OCTAHEDRON.tris[0]]).T
    initial_state = matrix_times_array_of_vectors(rot_mat, verts_og)[..., 0:2]
    #initial_state = normalize_func(initial_state)

    t0 = perf_counter()
    verts_new = gradient_descent(cost_grad_func_maker(0.2),
                                 initial_state,
                                 iteration_count=it_count,
                                 #normalize_func=normalize_func)
                                 grad_tolerance=1e-2,
                                 )
    verts_new = gradient_descent(cost_grad_func_maker(0.001),
                                 initial_state,
                                 iteration_count=it_count,
                                 #normalize_func=normalize_func)
                                 grad_tolerance=1e-5,
                                 )
    t1 = perf_counter()
    print(f"{t1 - t0:.2f} seconds")
    
    scale_factors = np.array([
        tri_area_plane(*verts_new[tri])/tri_area_plane(*verts_og[tri])
        for tri in tris])
    print(np.min(scale_factors), np.max(scale_factors))
    set_up_plot(0.85, 0.1)
    plot_mesh(Mesh(verts=verts_new, tris=tris))
    """
    lines = octant_graticule(18)
    for line in lines:
        def to_new_mesh(v):
            return point_old_mesh_to_new(v, verts_og, verts_new, tris)
        line_new = np.apply_along_axis(to_new_mesh, 1, line)
        plot_curve(line_new)
    """
    return Mesh(verts=verts_new, tris=tris)


def octant_graticule(n, resolution=0.005):
    lines = []
    for i in range(n + 1):
        lon = np.pi/2 * i/n
        start = np.array([lon, 0])
        end = np.array([lon, np.pi/2])
        num_points = int(np.ceil(np.pi/(2*resolution))) + 1
        line_lonlat = np.linspace(start, end, num_points)
        line = np.apply_along_axis(lonlat_to_cartes, 1, line_lonlat)
        lines.append(line)
    for i in range(n):
        lat = np.pi/2 * i/n
        start = np.array([0, lat])
        end = np.array([np.pi/2, lat])
        num_points = int(np.ceil(np.pi/(2*resolution) * np.cos(lat))) + 1
        line_lonlat = np.linspace(start, end, num_points)
        line = np.apply_along_axis(lonlat_to_cartes, 1, line_lonlat)
        lines.append(line)
    return lines


def plot_curve(curve):
    curve_2d = curve[:, 0:2]
    xs, ys = curve_2d.T
    plt.plot(xs, ys, c="0", linewidth=0.8)


def set_up_plot(lims=1.1, y_offset=0):
    plt.cla()
    plt.gca().set_aspect("equal")
    plt.gca().set_xlim(-lims, lims)
    plt.gca().set_ylim(-lims + y_offset, lims + y_offset)
    plt.show()


def plot_mesh(mesh):
    for tri in mesh.tris:
        a, b, c = mesh.verts[tri]
        if a.shape[0] == 3:
            if np.cross(b-a, c-a)[2] <= 0:
                continue
        a2d, b2d, c2d = a[0:2], b[0:2], c[0:2]
        xs, ys = np.column_stack([a2d, b2d, c2d, a2d])
        plt.plot(xs, ys)


def main():
    plt.ion()


if __name__ == "__main__":
    main()
