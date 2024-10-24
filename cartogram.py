import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple
import json
from time import perf_counter
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon


rng = np.random.default_rng()


with open("ne_50m_admin_0_countries_lakes_FIXED.json", "r") as f:
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
			"Romania": 19257520, "Russia": 144104080, "Rwanda": 12952210, "Samoa": 198410, "San Marino": 33940, "Sao Tome and Principe": 219160,
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


def norm_flat(v):
    return np.sqrt(dot_flat(v, v))


def dot_nzd_flat(v0, v1):
    return dot_flat(v0, v1) / (norm_flat(v0) * norm_flat(v1))


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
        if (cross_a > max_cross_a + TOLERANCE
                or cross_a < min_cross_a - TOLERANCE):
            return None
    if not infinite_b:
        if (cross_b > max_cross_b + TOLERANCE
                or cross_b < min_cross_b - TOLERANCE):
            return None
    return point


def line_intersection_sphere(a0, a1, b0, b1, infinite_b=False):
    assert a0.shape[0] == 3
    point_cross = np.cross(np.cross(a0, a1), np.cross(b0, b1))
    if (np.abs(point_cross) < TOLERANCE).all():     # if close to parallel
        return None
    point = nzd(point_cross)
    a_mid = nzd(a0 + a1)
    b_mid = nzd(b0 + b1)
    point *= np.sign(a_mid @ point)
    if vec_norm(point - a_mid) > vec_norm(a0 - a_mid) + TOLERANCE:
        return None
    if not infinite_b:
        if vec_norm(point - b_mid) > vec_norm(b0 - b_mid) + TOLERANCE:
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


def cartes_to_lonlat(v):
    x, y, z = v
    return np.array([np.arctan2(y, x), np.arcsin(z)])


def hammer_projection(lonlat):
    lon, lat = lonlat
    x = (2**1.5 * np.cos(lat) * np.sin(lon/2)
         / np.sqrt(1 + np.cos(lat) * np.cos(lon/2)))
    y = (2**0.5 * np.sin(lat)
         / np.sqrt(1 + np.cos(lat) * np.cos(lon/2)))
    return np.array([x, y])


def equal_earth_derivs(x):
    t = np.arcsin(np.sqrt(3)/2 * np.sin(x))
    sx, cx, st, ct = np.sin(x), np.cos(x), np.sin(t), np.cos(t)
    dtdx = np.sqrt(3)/2 * np.cos(x) / np.cos(t)
    a1 = 1.340264
    a2 = -0.081106
    a3 = 0.000893
    a4 = 0.003796
    f = a4 * t**9 + a3 * t**7 + a2 * t**3 + a1 * t
    fi = 9*a4 * t**8 + 7*a3 * t**6 + 3*a2 * t**2 + a1
    fii = 72*a4 * t**7 + 42*a3 * t**5 + 6*a2 * t
    fiii = 504*a4 * t**6 + 210*a3 * t**4 + 6*a2
    fp = fi * dtdx
    fpp = (fii * dtdx**2
           + fi * np.sqrt(3)/2
             * (-sx * ct + np.sqrt(3)/2 * cx**2 * st/ct) / ct**2)
    fppp = (fiii * dtdx**3
            + fii
              * (3/2 * (-cx * sx * ct + cx**2 * st * dtdx) / ct**3
                 + dtdx * np.sqrt(3)/2
                   * (-sx * ct + np.sqrt(3)/2 * cx**2 * st/ct) / ct**2)
            + fi * np.sqrt(3)/2
              * ((-cx * ct + sx * st * dtdx - np.sqrt(3) * cx * sx * st/ct
                  + cx**2 / ct**2 * np.sqrt(3)/2 * dtdx) * ct
                 + (-sx * ct + np.sqrt(3)/2 * cx**2 * st/ct) * 2 * st * dtdx)
              / ct**3)
    return f, fp, fpp, fppp


def sinusoidal_derivs(x):
    return x, 1 + 0*x, 0*x, 0*x


def mollweide_derivs(x):
    a1 = 1.11039784780
    a3 = -0.0662943858992
    a5 = -0.0207086475913
    a7 = 0.0312619523127
    a9 = -0.0261916651107
    a11 = 0.0102240520704
    a13 = -0.00156787281335
    f = (a1 * x + a3 * x**3 + a5 * x**5 + a7 * x**7 + a9 * x**9
         + a11 * x**11 + a13 * x**13)
    fp = (a1 + 3*a3 * x**2 + 5*a5 * x**4 + 7*a7 * x**6 + 9*a9 * x**8
          + 11*a11 * x**10 + 13*a13 * x**12)
    fpp = (6*a3 * x + 20*a5 * x**3 + 42*a7 * x**5 + 72*a9 * x**7
           + 110*a11 * x**9 + 156*a13 * x**11)
    fppp = (6*a3 + 60*a5 * x**2 + 210*a7 * x**4 + 504*a9 * x**6
            + 990*a11 * x**8 + 1716*a13 * x**10)
    return f, fp, fpp, fppp


def pseudocylindrical(derivs_func, lonlat):
    l, x = lonlat
    f, fp, _, _ = derivs_func(x)
    return np.array([l * np.cos(x) / fp, f])


def pseudocyl_blurred_jacobian_grad(derivs_func, v, eps):
    l, x = cartes_to_lonlat(v)
    f, fp, fpp, fppp = derivs_func(x)
    def q(s):
        return (1 - np.sin(np.pi/2 * np.cos(np.pi * s))) / 2
    def qp(s):
        return (np.pi**2 / 4 * np.cos(np.pi/2 * np.cos(np.pi * s))
                * np.sin(np.pi * s))
    sx, cx = np.sin(x), np.cos(x)
    g = (-sx * fp - cx * fpp) / fp**2
    gp = ((-cx * fp**2 - cx * fppp * fp + 2 * sx * fpp * fp + 2 * cx * fpp**2)
          / fp**3)
    l_pos = np.where(l >= 0, l, l + 2*np.pi)
    k = 2/np.pi * np.arccos(eps / np.pi)
    eps_sec = eps / np.cos(k * x)
    on_antimer = np.abs(l) > np.pi - eps_sec
    on_pole = np.abs(x) > np.pi/2 - eps
    a = (l_pos - 2 * np.pi * q((l_pos - np.pi) / (2 * eps_sec) + 1/2)) * g
    zero = np.zeros(x.shape)
    one = np.ones(x.shape)
    id2 = np.identity(2)[..., np.newaxis]
    J1 = np.array([[1 / fp, np.where(on_antimer, a, l * g)],
                   [zero, fp]])
    H = np.where(on_pole,
                 J1 * q((np.pi/2 - np.abs(x)) / eps)
                 + id2 * (1 - q((np.pi/2 - np.abs(x)) / eps)),
                 J1)
    b = (1 - 2 * np.pi
         * qp((l_pos - np.pi) / (2 * eps_sec) + 1/2)
         / (2 * eps_sec)) * g
    c = (np.pi * k
         * qp((l_pos - np.pi) / (2 * eps_sec) + 1/2)
         * (l_pos - np.pi) / eps * np.sin(k * x) * g
         + (l_pos - 2 * np.pi * q((l_pos - np.pi) / (2 * eps_sec) + 1/2)) * gp)
    dJ1dl = np.array([[zero, np.where(on_antimer, b, g)],
                      [zero, zero]])
    dJ1dx = np.array([[-fpp / fp**2, np.where(on_antimer, c, l * gp)],
                      [zero, fpp]])
    dHdl = np.where(on_pole, dJ1dl * q((np.pi/2 - np.abs(x)) / eps), dJ1dl)
    dHdx = np.where(on_pole,
                    dJ1dx * q((np.pi/2 - np.abs(x)) / eps)
                    - (J1 - id2) * qp((np.pi/2 - np.abs(x)) / eps)
                    * np.sign(x) / eps,
                    dJ1dx)
    # "tspx" is x in the tangent space, "x" is phi (latitude). Sorry
    dHdtspx = (dHdl / np.cos(x)
               + np.tan(x) * mul_veczd(H, np.array([[zero, one],
                                                    [-one, zero]])))

    is_pole = np.abs(x) > np.pi/2 - TOLERANCE
    if is_pole.any():
        H = np.where(is_pole, np.array([[one, zero], [zero, one]]), H)
        dHdtspx = np.where(is_pole, np.array([[zero, zero], [zero, zero]]),
                           dHdtspx)
        dHdx = np.where(is_pole, np.array([[zero, zero], [zero, zero]]), dHdx)
    return ValueGrad(value=H, grad=np.array([dHdtspx, dHdx]))


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


def subdivide_octahedron_interrupted(n, shift_degrees=0, return_boundary=True):
    mesh = OCTAHEDRON
    # Subdivide each mesh triangle. Store new vertices and triangles.
    verts_og, tris_og = mesh.verts, mesh.tris
    verts_subdiv_list = []
    tris_subdiv_list = []
    verts_per_og_tri = ((n + 1) * (n + 2)) // 2
    verts_og_centers_list = []
    for i, tri in enumerate(tris_og):
        verts_new, tris_new = subdivide_tri_sphere(*verts_og[tri], n)
        verts_subdiv_list.append(verts_new)
        tris_subdiv_list.append(tris_new + i*verts_per_og_tri)

        a, b, c = verts_og[tri]
        center = (a + b + c) / 3
        verts_og_centers_list += [center] * verts_per_og_tri
    verts = np.concatenate(verts_subdiv_list)
    tris = np.concatenate(tris_subdiv_list)
    verts_og_centers = np.array(verts_og_centers_list)
    verts_proj_list = []
    for i, v in enumerate(verts):
        lonlat = cartes_to_lonlat(v)
        if abs(abs(lonlat[0]) - np.pi) < TOLERANCE:
            lonlat[0] = np.pi * np.sign(verts_og_centers[i, 1])
        verts_proj_list.append(hammer_projection(lonlat))
    verts_proj = np.array(verts_proj_list)

    edge_vert_ixs = tri_edge_vert_indices(n)
    ixs_to_check = np.concatenate( # edge indices to check for duplicates
        [edge_vert_ixs + i*verts_per_og_tri for i in range(tris_og.shape[0])])
    first_seen_ixs = []
    old_ixs_to_new = np.arange(verts.shape[0])
    for ix in ixs_to_check:
        for seen_ix in first_seen_ixs:
            if vec_norm(verts_proj[ix] - verts_proj[seen_ix]) < TOLERANCE:
                old_ixs_to_new[ix] = seen_ix
                break
        else: # nobreak
            first_seen_ixs.append(ix)

    ixs_unique = np.unique(old_ixs_to_new)
    verts_final = verts[ixs_unique]
    verts_proj_final = verts_proj[ixs_unique]
    new_ixs_to_final = np.arange(verts.shape[0])
    for i, val in enumerate(ixs_unique):
        new_ixs_to_final[val] = i
    old_ixs_to_final = new_ixs_to_final[old_ixs_to_new]
    tris_final = old_ixs_to_final[tris]

    if return_boundary:
        boundary = np.logical_and(np.abs(verts_final[:, 1]) < TOLERANCE,
                                  verts_final[:, 0] < TOLERANCE)

    shift = np.deg2rad(shift_degrees)
    shift_mat = np.array([[np.cos(shift), -np.sin(shift), 0],
                          [np.sin(shift), np.cos(shift), 0],
                          [0, 0, 1]])
    verts_final = matrix_times_array_of_vectors(shift_mat, verts_final)
    if return_boundary:
        return (Mesh(verts=verts_final, tris=tris_final),
                verts_proj_final,
                boundary)
    return Mesh(verts=verts_final, tris=tris_final), verts_proj_final


def subdivide_mesh_selected(mesh, tris_to_cut):
    def tup(list_):
        return tuple(sorted(list_))

    verts_new = list(mesh.verts)
    tris_new = []
    edge_ixs_dict = {}
    
    def add_edge(p0, p1):
        edge = tup([p0, p1])
        if edge not in edge_ixs_dict:
            midpoint = (mesh.verts[p0] + mesh.verts[p1]) / 2
            if midpoint.shape[0] == 3:
                midpoint = nzd(midpoint)
            verts_new.append(midpoint)
            edge_ixs_dict[edge] = len(verts_new) - 1            
            
    for ix in tris_to_cut:
        tri = mesh.tris[ix]
        add_edge(tri[0], tri[1])
        add_edge(tri[1], tri[2])
        add_edge(tri[2], tri[0])
    for tri in mesh.tris:
        a, b, c = tri
        d = edge_ixs_dict.get(tup([a, b]))
        e = edge_ixs_dict.get(tup([b, c]))
        f = edge_ixs_dict.get(tup([c, a]))
        d_fr = d is not None
        e_fr = e is not None
        f_fr = f is not None
        if d_fr and e_fr and f_fr:
            tris_new += [[a, d, f], [b, e, d], [c, f, e], [d, e, f]]
        elif d_fr and e_fr and not f_fr:
            tris_new += [[a, d, c], [b, e, d], [c, d, e]]
        elif d_fr and not e_fr and f_fr:
            tris_new += [[a, d, f], [b, f, d], [c, f, b]]
        elif d_fr and not e_fr and not f_fr:
            tris_new += [[a, d, c], [b, c, d]]
        elif not d_fr and e_fr and f_fr:
            tris_new += [[a, b, e], [a, e, f], [c, f, e]]
        elif not d_fr and e_fr and not f_fr:
            tris_new += [[a, b, e], [a, e, c]]
        elif not d_fr and not e_fr and f_fr:
            tris_new += [[a, b, f], [b, c, f]]
        else:
            tris_new += [[a, b, c]]
    return Mesh(verts=np.array(verts_new), tris=np.array(tris_new))


def mesh_edges_dict(mesh):
    edges_dict = {}
    for i, tri in enumerate(mesh.tris):
        for j in range(3):
            pt0, pt1 = tri[j], tri[(j+1) % 3]
            edge = min(pt0, pt1), max(pt0, pt1)
            if edge not in edges_dict:
                edges_dict[edge] = []
            edges_dict[edge].append(i)
            if len(edges_dict[edge]) == 2:
                edges_dict[edge] = np.array(edges_dict[edge])
    return edges_dict


def mesh_tri_neighbors_list(mesh):
    edges_dict = mesh_edges_dict(mesh)
    neighbs_list = []
    for i in range(mesh.tris.shape[0]):
        neighbs_list.append([])
    for tri_pair in edges_dict.values():
        if len(tri_pair) == 1:
            continue    # edge is on interruption, so borders only one tri
        tri0, tri1 = tri_pair
        neighbs_list[tri0].append(tri1)
        neighbs_list[tri1].append(tri0)
    return neighbs_list


def clamp_to_tangent_space(a, b, c):
    center = nzd((a + b + c) / 3)
    a1 = a - (a @ center) * center
    b1 = b - (b @ center) * center
    c1 = c - (c @ center) * center
    return a1, b1, c1


def tangent_space_matrix(a, b, c,
                         clamp_to_sphere=False,
                         clamp_to_tan_space=False):
    dim = a.shape[0]
    assert dim == 2 or dim == 3
    if dim == 2:
        return np.identity(2)

    a1, b1, c1 = a, b, c
    if clamp_to_sphere and clamp_to_tan_space:
        a1, b1, c1 = clamp_to_tangent_space(a, b, c)
    assert vec_norm(np.cross(b-a, c-a)) >= TOLERANCE
    t = nzd(b1 - a1)
    n = nzd(np.cross(b1-a1, c1-a1))
    if clamp_to_sphere and n @ (a + b + c) < 0:
        n = -n
    s = np.cross(n, t)
    return np.column_stack([t, s, n])


# returns matrix that takes the vecs [1,0], [0,1] to b-a, c-a in tangent space
def matrix_basis_vecs_to_tri(a, b, c, clamp_to_sphere=False):
    dim = a.shape[0]
    assert dim == 2 or dim == 3
    if dim == 2:
        return np.column_stack([b-a, c-a])

    tan_space_mat = tangent_space_matrix(a, b, c,
                                         clamp_to_sphere,
                                         clamp_to_tan_space=clamp_to_sphere)
    tan_space_inv = tan_space_mat.T
    a1, b1, c1 = a, b, c
    if clamp_to_sphere:
        a1, b1, c1 = clamp_to_tangent_space(a, b, c)
    return tan_space_inv[0:2] @ np.column_stack([b1-a1, c1-a1])


def minimize(
        cost_grad_func, # returns ValueGrad
        initial_state,
        *,
        normalize_func=lambda x: x,
        iteration_count=100,
        memory=10,
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

    cost_grad = cost_grad_func(x)
    for i in range(iteration_count):
        if i % 100 == 0:
            print(f"iter {i} cost {cost_grad.value:.5f} grad {norm_flat(cost_grad.grad):.5f}")
        g = cost_grad.grad
        if (np.abs(g) < grad_tolerance).all():
            return x
        search_dir = -H(len(s), g)
        search_dir_local = -H_0_scale * g
        descent_dot = dot_nzd_flat(g, search_dir)
        if descent_dot >= 0 or H_0_scale < TOLERANCE:
            s = []
            y = []
            rho = []
            H_0_scale = 1
            search_dir = -H(len(s), g)
            search_dir_local = -H_0_scale * g
            descent_dot = dot_nzd_flat(g, search_dir)
        
        x_new, cost_grad_new = line_search(cost_grad_func,
                                           x,
                                           cost_grad,
                                           search_dir,
                                           search_dir_local,
                                           normalize_func=normalize_func)
        g_new = cost_grad_new.grad

        #print(f"descent dot {descent_dot:.5f}")
        #descent_dot = dot_nzd_flat(g, x_new - x)
        #print(f"actual descent dot {descent_dot:.5f}")
        
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
        #if learn_rate < 1e-8:
        #    return x
    return x


def line_search(cost_grad_func,
                state,
                initial_cost_grad,
                search_dir,
                search_dir_local,
                *,
                normalize_func=lambda x: x,
                tau=0.5,
                c=0.1):
    #print("START LINE SEARCH")
    learn_rate = 1
    step_new = learn_rate * (learn_rate * search_dir
                             + (1 - learn_rate) * search_dir_local)
    state_new = normalize_func(state + step_new)
    cost_grad_new = cost_grad_func(state_new)

    def is_okay(cost_grad, step):
        grad_inc_threshold = 5
        return ((cost_grad.value - initial_cost_grad.value
                 <= c * dot_flat(initial_cost_grad.grad, step) + TOLERANCE)
                and (norm_flat(cost_grad.grad)
                     < grad_inc_threshold * norm_flat(initial_cost_grad.grad)))
    
    if is_okay(cost_grad_new, step_new): # starting step is small enough
        #print(f"{learn_rate} is small enough")
        return state_new, cost_grad_new
    # starting step is too big
    #print(f"{learn_rate} is too big")
    while True:
        learn_rate *= tau   # make the step smaller
        step_new = learn_rate * (learn_rate * search_dir
                                 + (1 - learn_rate) * search_dir_local)
        state_new = normalize_func(state + step_new)
        cost_grad_new = cost_grad_func(state_new)
        if is_okay(cost_grad_new, step_new):  # step is now small enough
            #print(f"{learn_rate} is small enough")
            return state_new, cost_grad_new


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
            return point_old_tri_to_new(v_plane,
                                        *verts_old[tri],
                                        *verts_new[tri])


def mesh_box_dict(mesh, resolution):
    a = mesh.verts[mesh.tris[:, 0]].T
    b = mesh.verts[mesh.tris[:, 1]].T
    c = mesh.verts[mesh.tris[:, 2]].T
    assert a.shape[0] == 3
    eps = np.max(np.array([dot_veczd(b-a, b-a),
                           dot_veczd(c-b, c-b),
                           dot_veczd(a-c, a-c)]), axis=0)
    box_min = np.min(np.array([a, b, c]), axis=0) - eps
    box_max = np.max(np.array([a, b, c]), axis=0) + eps
    box_ix_min = np.floor(box_min / resolution).astype(int)
    box_ix_max = np.ceil(box_max / resolution).astype(int)
    box_dict = {}
    for i in range(a.shape[1]):
        for jx in range(box_ix_min[0, i], box_ix_max[0, i]):
            for jy in range(box_ix_min[1, i], box_ix_max[1, i]):
                for jz in range(box_ix_min[2, i], box_ix_max[2, i]):
                    if (jx, jy, jz) not in box_dict:
                        box_dict[(jx, jy, jz)] = []
                    box_dict[(jx, jy, jz)].append(i)
    return box_dict


def polys_old_mesh_to_new(polys, mesh_old, mesh_new):
    assert mesh_old.verts.shape[0] == mesh_new.verts.shape[0]
    assert mesh_old.tris.shape[0] == mesh_new.tris.shape[0]
    resolution = 0.01
    box_dict = mesh_box_dict(mesh_old, resolution)
    polys_new = []
    for verts in polys:
        verts_new = []
        for v in verts:
            v_ix = np.floor(v / resolution).astype(int)
            v_box_tris = box_dict[(v_ix[0], v_ix[1], v_ix[2])]
            v_tris = mesh_old.tris[v_box_tris]
            for tri in v_tris:
                if is_point_inside_tri(v, *mesh_old.verts[tri]):
                    v_plane = point_to_tri_plane(v, *mesh_old.verts[tri])
                    v_new = point_old_tri_to_new(v_plane,
                                                 *mesh_old.verts[tri],
                                                 *mesh_new.verts[tri])
                    if v_new.shape[0] == 3:
                        v_new = nzd(v_new)
                    verts_new.append(v_new)
                    break
            else:   # no tri containing v is found
                raise Exception
        polys_new.append(np.array(verts_new))
    return polys_new


def interrupt_polygon_sphere(polygon, is_closed, b0, b1, infinite_b=False):
    output_list = [[]]
    is_interrupted = False
    poly = list(polygon)
    if is_closed:
        poly.append(poly[0])
    for i in range(len(poly) - 1):
        v0 = poly[i]
        v1 = poly[i + 1]
        p = line_intersection_sphere(v0, v1, b0, b1, infinite_b)
        if p is None:
            output_list[-1].append(v0)
        else:
            is_interrupted = True
            p0 = nzd(p + 0.001 * (v0 - p))
            p1 = nzd(p + 0.001 * (v1 - p))
            output_list[-1] += [v0, p0]
            output_list.append([p1])
    if not is_closed:
        output_list[-1].append(poly[-1])
    elif is_interrupted:
        start = output_list.pop(0)
        output_list[-1] += start
    is_closed_new = len(output_list) * [is_closed and not is_interrupted]
    for i in range(len(output_list)):
        output_list[i] = np.array(output_list[i])
    return output_list, is_closed_new


def interrupt_polygon_list(poly_list, is_closed_list,
                           b0, b1, infinite_b=False):
    assert len(poly_list) == len(is_closed_list)
    poly_list_new = []
    is_closed_list_new = []
    for i, poly in enumerate(poly_list):
        polys_new, is_closed_new = interrupt_polygon_sphere(poly,
                                        is_closed_list[i],
                                        b0, b1, infinite_b)
        poly_list_new += polys_new
        is_closed_list_new += is_closed_new
    return poly_list_new, is_closed_list_new


def interrupt_polygon_list_antimeridian(poly_list,
                                        is_closed_list,
                                        shift_degrees=0):
    ang = np.deg2rad(shift_degrees)
    interrupt_point = np.array([-np.cos(ang), -np.sin(ang), 0])
    poly_list_new, is_closed_list_new = interrupt_polygon_list(poly_list,
                                                is_closed_list,
                                                np.array([0, 0, 1]),
                                                interrupt_point)
    poly_list_new, is_closed_list_new = interrupt_polygon_list(poly_list_new,
                                                is_closed_list_new,
                                                interrupt_point,
                                                np.array([0, 0, -1]))
    return poly_list_new, is_closed_list_new


def clip_polygon_list_hemisphere(poly_list, is_closed_list):
    poly_list_new, is_closed_list_new = interrupt_polygon_list(poly_list,
                                                is_closed_list,
                                                np.array([1, 0, 0]),
                                                np.array([0, 1, 0]),
                                                infinite_b=True)
    poly_list_clipped = []
    is_closed_list_clipped = []
    for i, poly in enumerate(poly_list_new):
        if np.max(poly[:, 2]) > 0:
            poly_list_clipped.append(poly)
            is_closed_list_clipped.append(is_closed_list_new[i])
    return poly_list_clipped, is_closed_list_clipped


def poly_list_from_borders(borders_data_flat):
    poly_list = []
    for polys in borders_data_flat.values():
        poly_list += [lonlat_to_cartes(poly.T).T for poly in polys]
    is_closed_list = len(poly_list) * [True]
    return poly_list, is_closed_list


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


def tri_scales_blurred(mesh,
                       portions,
                       region_scales_intended,
                       num_blurs=256):
    neighbs = mesh_tri_neighbors_list(mesh)
    land_portions = np.sum(portions, axis=1)
    is_water = land_portions < TOLERANCE
    tri_scales_list = []
    for i in range(mesh.tris.shape[0]):
        if is_water[i]:
            tri_scales_list.append(1)
        else:
            scale_avg = (np.sum(portions[i] * region_scales_intended)
                         / land_portions[i])
            tri_scales_list.append(scale_avg)
    tri_scales = np.array(tri_scales_list)
    for i in range(num_blurs):
        tri_scales_new = tri_scales.copy()
        for j, neighb_ixs in enumerate(neighbs):
            if not is_water[j]:
                continue
            ixs_list = [j] + neighb_ixs
            vicinity_scales = np.array([tri_scales[k] for k in ixs_list])
            tri_scales_new[j] = np.mean(vicinity_scales)
        tri_scales = tri_scales_new
    return tri_scales


def matrix_basis_vecs_to_tri_veczd(verts, tris):
    num_tris = tris.shape[0]
    a = verts[tris[:, 0]].T
    b = verts[tris[:, 1]].T
    c = verts[tris[:, 2]].T
    if a.shape[0] == 2:
        tan_space_mats = np.array([np.identity(2)] * num_tris)
        mats_basis_vecs_to_tri = np.array([b-a, c-a]).transpose(1, 0, 2)
        return tan_space_mats, mats_basis_vecs_to_tri
    if a.shape[0] == 3:
        assert (norm_veczd(cross_veczd(b-a, c-a)) > TOLERANCE).all()
        t = nzd_veczd(b - a)
        n = nzd_veczd(cross_veczd(b-a, c-a))
        s = cross_veczd(n, t)
        tan_space_mats = np.array([t, s, n]).transpose(1, 0, 2)
        tan_space_invs = tan_space_mats.transpose(1, 0, 2)
        mats_basis_vecs_to_tri = mul_veczd(tan_space_invs[0:2],
                                    np.array([b-a, c-a]).transpose(1, 0, 2))
        return tan_space_mats, mats_basis_vecs_to_tri
    raise ValueError


def matrix_basis_vecs_to_tri_sphere_veczd(verts, tris):
    num_tris = tris.shape[0]
    a0 = verts[tris[:, 0]].T
    b0 = verts[tris[:, 1]].T
    c0 = verts[tris[:, 2]].T
    if a0.shape[0] != 3:
        raise ValueError
    centers = nzd_veczd((a0 + b0 + c0) / 3)
    a = a0 - dot_veczd(a0, centers) * centers
    b = b0 - dot_veczd(b0, centers) * centers
    c = c0 - dot_veczd(c0, centers) * centers
    east = cross_veczd(np.array([[0], [0], [1]]), centers)
    t = np.where(norm_veczd(east) > TOLERANCE,
                 nzd_veczd(east),
                 np.array([[1], [0], [0]]))
    n = centers
    s = cross_veczd(n, t)
    tan_space_mats = np.array([t, s, n]).transpose(1, 0, 2)
    tan_space_invs = tan_space_mats.transpose(1, 0, 2)
    mats_basis_vecs_to_tri = mul_veczd(tan_space_invs[0:2],
                                np.array([b-a, c-a]).transpose(1, 0, 2))
    return tan_space_mats, mats_basis_vecs_to_tri


def dot_veczd(vecs0, vecs1):
    return np.sum(vecs0 * vecs1, axis=0)


def norm_veczd(vecs):
    return np.sqrt(np.sum(vecs * vecs, axis=0))


def nzd_veczd(vecs):
    return vecs / norm_veczd(vecs)


def cross_veczd(vecs0, vecs1):
    if vecs0.shape[0] == 2 and vecs1.shape[0] == 2:
        return vecs0[0] * vecs1[1] - vecs0[1] * vecs1[0]
    elif vecs0.shape[0] == 3 and vecs1.shape[0] == 3:
        return np.array([vecs0[1] * vecs1[2] - vecs0[2] * vecs1[1],
                         vecs0[2] * vecs1[0] - vecs0[0] * vecs1[2],
                         vecs0[0] * vecs1[1] - vecs0[1] * vecs1[0]])
    raise ValueError


def det_2d_veczd(mats):
    return mats[0, 0] * mats[1, 1] - mats[0, 1] * mats[1, 0]


def inv_2d_veczd(mats):
    return np.array([[mats[1, 1], -mats[0, 1]],
                     [-mats[1, 0], mats[0, 0]]]) / det_2d_veczd(mats)


def mul_veczd(mats0, mats1):
    if len(mats0.shape) == 2:
        m0 = mats0[np.newaxis].transpose(2, 0, 1)
    elif len(mats0.shape) == 3:
        m0 = mats0.transpose(2, 0, 1)
    else:
        raise ValueError
    if len(mats1.shape) == 2:
        m1 = mats1[:, np.newaxis].transpose(2, 0, 1)
    elif len(mats1.shape) == 3:
        m1 = mats1.transpose(2, 0, 1)
    else:
        raise ValueError
    m2 = (m0 @ m1).transpose(1, 2, 0)
    if len(mats0.shape) == 2 and len(mats1.shape) == 3:
        m2 = m2[0]
    elif len(mats0.shape) == 3 and len(mats1.shape) == 2:
        m2 = m2[:, 0]
    elif len(mats0.shape) == 2 and len(mats1.shape) == 2:
        m2 = m2[0, 0]
    return m2


def tri_det_dist_value_grads_veczd(G0, G, H=None, H_grad=None):
    # H_grad is of the form [1/cos(lat) * dHdlon, dHdlat]
    hybrid = H is not None and H_grad is not None
    if hybrid:
        H_grad_pts = 1/3 * np.array([H_grad, H_grad, H_grad])
    num_tris = G.shape[-1]
    G0inv = inv_2d_veczd(G0)
    G_grad = np.array([[[[-1, -1], [0, 0]], [[0, 0], [-1, -1]]],
                       [[[1, 0], [0, 0]], [[0, 0], [1, 0]]],
                       [[[0, 1], [0, 0]], [[0, 0], [0, 1]]]])[..., np.newaxis]
    E = mul_veczd(G, G0inv)
    E_grad = np.empty((3, 2, 2, 2, num_tris))
    for i in range(3):
        for j in range(2):
            E_grad[i, j] = mul_veczd(G_grad[i, j], G0inv)
    D = det_2d_veczd(E)
    D_grad = (E_grad[:, :, 0, 0] * E[1, 1]
              + E[0, 0] * E_grad[:, :, 1, 1]
              - E_grad[:, :, 0, 1] * E[1, 0]
              - E[0, 1] * E_grad[:, :, 1, 0])
    if hybrid:
        E1 = mul_veczd(H, E)
        E1_grad = np.empty((3, 2, 2, 2, num_tris))
        for i in range(3):
            for j in range(2):
                E1_grad[i, j] = (mul_veczd(H_grad_pts[i, j], E)
                                 + mul_veczd(H, E_grad[i, j]))
        D1 = det_2d_veczd(E1)
        D1_grad = (E1_grad[:, :, 0, 0] * E1[1, 1]
                  + E1[0, 0] * E1_grad[:, :, 1, 1]
                  - E1_grad[:, :, 0, 1] * E1[1, 0]
                  - E1[0, 1] * E1_grad[:, :, 1, 0])
    else:
        E1, E1_grad, D1, D1_grad = E, E_grad, D, D_grad
    F1 = np.sum(E1 * E1, axis=(0, 1))
    F1_grad = 2 * (E1[0, 0] * E1_grad[:, :, 0, 0]
                  + E1[0, 1] * E1_grad[:, :, 0, 1]
                  + E1[1, 0] * E1_grad[:, :, 1, 0]
                  + E1[1, 1] * E1_grad[:, :, 1, 1])
    dist = F1/D1 - 2
    dist_grad = (F1_grad*D1 - F1*D1_grad) / (D1*D1)
    det_value_grad = ValueGrad(value=D, grad=D_grad)
    dist_value_grad = ValueGrad(value=dist, grad=dist_grad)
    return det_value_grad, dist_value_grad


def cartogram(mesh,
              portions,
              pop_array,
              max_iterations,
              *,
              sphere_first=False,
              hybrid=False,
              fix_antimer=False,
              proj_derivs=equal_earth_derivs,
              initial_verts=None,
              boundary=None):
    if (hybrid or fix_antimer) and not sphere_first:
        raise ValueError
    G0 = matrix_basis_vecs_to_tri_veczd(mesh.verts, mesh.tris)[1]
    M0 = 1/2 * det_2d_veczd(G0)
    tri_region_areas_og = M0[:, np.newaxis] * portions
            # absolute area of each region falling in each tri
    region_areas_og = np.sum(tri_region_areas_og, axis=0)
    world_pop_density = np.sum(pop_array) / np.sum(region_areas_og)
    region_areas_intended = pop_array / world_pop_density
    region_scales_intended = region_areas_intended / region_areas_og
    land_portions = np.sum(portions, axis=1)
    A = 1 + 0*tri_scales_blurred(mesh,
                           portions,
                           region_scales_intended,
                           num_blurs=256)
    for i, v in enumerate(mesh.verts):
        if np.abs(v[0]) < TOLERANCE and np.abs(v[1]) < TOLERANCE and v[2] > 0:
            npole_ix = i
            break
    if boundary is not None and not sphere_first:
        quad0 = np.logical_and(boundary, np.logical_and(
                               initial_verts[:, 0] > TOLERANCE,
                               initial_verts[:, 1] > TOLERANCE))
        quad1 = np.logical_and(boundary, np.logical_and(
                               initial_verts[:, 0] < -TOLERANCE,
                               initial_verts[:, 1] > TOLERANCE))
        quad2 = np.logical_and(boundary, np.logical_and(
                               initial_verts[:, 0] < -TOLERANCE,
                               initial_verts[:, 1] < -TOLERANCE))
        quad3 = np.logical_and(boundary, np.logical_and(
                               initial_verts[:, 0] > TOLERANCE,
                               initial_verts[:, 1] < -TOLERANCE))
        pole0 = np.logical_and(boundary, np.logical_and(
                               np.abs(initial_verts[:, 0]) < TOLERANCE,
                               initial_verts[:, 1] > 0))
        pole1 = np.logical_and(boundary, np.logical_and(
                               np.abs(initial_verts[:, 0]) < TOLERANCE,
                               initial_verts[:, 1] < 0))
        pole0_ix = np.argmax(pole0)
        pole1_ix = np.argmax(pole1)

    def cost_grad_func_maker(weights_dist,
                             weights_area,
                             weight_boundary,
                             weights_antimer,
                             weight_error):

        def cost_grad_func(verts):
            if boundary is not None and not sphere_first:
                diffs_pole0 = verts[:, 0] - verts[pole0_ix, 0]
                diffs_pole1 = verts[:, 0] - verts[pole1_ix, 0]
                diffs_quad0 = np.where(quad0, diffs_pole0, np.inf)
                diffs_quad1 = np.where(quad1, -diffs_pole0, np.inf)
                diffs_quad2 = np.where(quad2, -diffs_pole1, np.inf)
                diffs_quad3 = np.where(quad3, diffs_pole1, np.inf)
                cost_boundary = 0
                cost_boundary_grad = np.zeros_like(verts)
                for diffs in [diffs_quad0, diffs_quad1,
                              diffs_quad2, diffs_quad3]:
                    if (diffs < TOLERANCE).any():
                        return ValueGrad(value=np.inf,
                                         grad=np.zeros_like(verts))
                    cost_boundary += np.sum(1 / diffs)
                cost_boundary_grad[:, 0] += -1 / (diffs_quad0 * diffs_quad0)
                cost_boundary_grad[:, 0] += 1 / (diffs_quad1 * diffs_quad1)
                cost_boundary_grad[:, 0] += 1 / (diffs_quad2 * diffs_quad2)
                cost_boundary_grad[:, 0] += -1 / (diffs_quad3 * diffs_quad3)
                cost_boundary_grad[pole0_ix, 0] = np.sum(
                        1 / (diffs_quad0 * diffs_quad0)
                        - 1 / (diffs_quad1 * diffs_quad1))
                cost_boundary_grad[pole1_ix, 0] = np.sum(
                        -1 / (diffs_quad2 * diffs_quad2)
                        + 1 / (diffs_quad3 * diffs_quad3))
                num_boundary_points = np.sum(np.where(boundary, 1, 0))
                cost_boundary /= num_boundary_points
                cost_boundary_grad /= num_boundary_points
            if fix_antimer:
                diffs_antimer = np.where(antimer,
                                         (verts - mesh.verts)[:, 1], 0)
                diffs_pole = np.where(north_pole,
                                      (verts - mesh.verts)[:, 0], 0)
                costs_antimer = diffs_antimer**2 + diffs_pole**2
                cost_antimer_grad = 2 * np.array([diffs_pole,
                                                  diffs_antimer,
                                                  0 * diffs_antimer])

            if sphere_first:
                tan_space_mats, G = matrix_basis_vecs_to_tri_sphere_veczd(
                    verts, mesh.tris)
            else:
                tan_space_mats, G = matrix_basis_vecs_to_tri_veczd(
                    verts, mesh.tris)
            M = 1/2 * det_2d_veczd(G)
            if (M < TOLERANCE).any():
                #print("cost inf")
                return ValueGrad(value=np.inf, grad=np.zeros_like(verts))
            tri_region_areas = M[:, np.newaxis] * portions
            region_areas = np.sum(tri_region_areas, axis=0)
            region_errors = ((region_areas - region_areas_intended)
                             / np.sqrt(region_areas_intended))
            cost_error = np.sum(region_errors * region_errors)
            H, H_grad = None, None
            if hybrid:
                centers = nzd_veczd(verts[mesh.tris[:, 0]].T
                                    + verts[mesh.tris[:, 1]].T
                                    + verts[mesh.tris[:, 2]].T)
                eps = 6 / np.sqrt(mesh.tris.shape[0])
                H, H_grad = pseudocyl_blurred_jacobian_grad(proj_derivs,
                                                            centers, eps)
            (D, D_grad), (dist, dist_grad) = tri_det_dist_value_grads_veczd(
                                                            G0, G, H, H_grad)
            #if rng.random() < 0.001:
            #    print(np.histogram(np.log(D), 20))
            costs_dist = M0 * dist
            costs_dist_grad = M0 * dist_grad
            costs_area = M0 * (D/A + A/D - 2)
            costs_area_grad = M0 * (D_grad / A - A*D_grad / (D*D))
            cost_error_grad = (2 * M0
                           * np.sum(portions.T
                                * region_errors[:, np.newaxis]
                                / np.sqrt(region_areas_intended)[:, np.newaxis],
                                axis=0)
                           * D_grad)
            cost = (np.sum(weights_dist * costs_dist)
                    + np.sum(weights_area * costs_area)
                    + weight_error * cost_error)
            tri_cost_grads_tan_space = (weights_dist * costs_dist_grad
                                        + weights_area * costs_area_grad
                                        + weight_error * cost_error_grad)
            if not sphere_first:
                tri_cost_grads = tri_cost_grads_tan_space
            else:
                tcg_tan_space_global_coords = (
                    tan_space_mats.transpose(2, 0, 1)[:, np.newaxis, :, 0:2]
                    @ tri_cost_grads_tan_space.transpose(
                        2, 0, 1)[..., np.newaxis]
                    )[..., 0].transpose(1, 2, 0)
                a = verts[mesh.tris[:, 0]].T
                b = verts[mesh.tris[:, 1]].T
                c = verts[mesh.tris[:, 2]].T
                a_grad = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                   [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                   [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
                                  )[..., np.newaxis]
                b_grad = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                   [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                   [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
                                  )[..., np.newaxis]
                c_grad = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                   [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                   [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
                                  )[..., np.newaxis]
                d = a + b + c
                d_grad = a_grad + b_grad + c_grad
                dn = nzd_veczd(d)
                tri_cost_grads = np.empty((3, 3, mesh.tris.shape[0]))
                for j in range(3):
                    for k in range(3):
                        dn_grad_jk = (d_grad[j, k] / norm_veczd(d)
                                      - d / norm_veczd(d)**3
                                      * dot_veczd(d, d_grad[j, k]))
                        at_grad_jk = (a_grad[j, k]
                                      - (dot_veczd(a_grad[j, k], dn)
                                         + dot_veczd(a, dn_grad_jk)) * dn
                                      + (1 - dot_veczd(a, dn)) * dn_grad_jk)
                        bt_grad_jk = (b_grad[j, k]
                                      - (dot_veczd(b_grad[j, k], dn)
                                         + dot_veczd(b, dn_grad_jk)) * dn
                                      + (1 - dot_veczd(b, dn)) * dn_grad_jk)
                        ct_grad_jk = (c_grad[j, k]
                                      - (dot_veczd(c_grad[j, k], dn)
                                         + dot_veczd(c, dn_grad_jk)) * dn
                                      + (1 - dot_veczd(c, dn)) * dn_grad_jk)
                        tri_cost_grads[j, k] = np.sum(
                            tcg_tan_space_global_coords * np.array([
                                at_grad_jk, bt_grad_jk, ct_grad_jk]),
                            axis=(0, 1))
                
            cost_grad = np.zeros_like(verts)
            for i in range(3):
                for j in range(verts.shape[-1]):
                    cost_grad[:, j] += np.histogram(
                        mesh.tris[:, i],
                        bins=np.arange(verts.shape[0] + 1),
                        weights=tri_cost_grads[i, j, :]
                    )[0]
            if boundary is not None and not sphere_first:
                cost += weight_boundary * cost_boundary
                cost_grad += weight_boundary * cost_boundary_grad
            if fix_antimer:
                cost += np.sum(weights_antimer * costs_antimer)
                cost_grad += (weights_antimer * cost_antimer_grad).T
                cost_grad[npole_ix] = 0
            if sphere_first:
                cost_grad -= (dot_veczd(cost_grad.T, verts.T) * verts.T).T
            #print(f"cost {cost:.5f} grad {norm_flat(cost_grad):.5f}")
            return ValueGrad(cost, cost_grad)

        return cost_grad_func

    def normalize_func(verts):
        if sphere_first:
            verts_out = nzd_veczd(verts.T).T
            if fix_antimer:
                verts_out[npole_ix] = [0, 0, 1]
            return verts_out
        return verts
    
    if initial_verts is not None:
        verts_new = initial_verts.copy()
    elif sphere_first:
        verts_new = mesh.verts.copy()
    else:
        raise ValueError

    is_water = land_portions < TOLERANCE
    weights_water = np.where(is_water, 0.01, 1)
    weights_pop = 0.2 + 0.8 * A
    antimer = np.logical_and(mesh.verts[:, 0] < TOLERANCE,
                             np.abs(mesh.verts[:, 1]) < TOLERANCE)
    north_pole = np.logical_and(antimer, 1 - mesh.verts[:, 2] < TOLERANCE)
    h = 0.5 + 0.5 * mesh.verts[:, 2]
    weights_antimer = np.where(h > 0.13, h**0.8, 0) / 97
    weights_antimer = np.where(h > (2 + np.sqrt(3)) / 4,
                               10 * weights_antimer,
                               weights_antimer)
    weights_dist = 1 * weights_water * weights_pop
    weights_area = 0.1 * weights_water * weights_pop
    weight_boundary = 1e-7
    weights_antimer = 100 * weights_antimer
    weight_error = 0
    """verts_new = minimize(cost_grad_func_maker(weights_dist,
                                              weights_area,
                                              weight_boundary,
                                              weights_antimer,
                                              weight_error),
                         verts_new,
                         iteration_count=max_iterations,
                         normalize_func=normalize_func,
                         grad_tolerance=1e-4)"""

    weights_dist = 1 * weights_water * weights_pop
    weights_area = 1 * weights_water * weights_pop
    weight_boundary = 1e-7
    weights_antimer *= 1
    weight_error = 0
    """verts_new = minimize(cost_grad_func_maker(weights_dist,
                                              weights_area,
                                              weight_boundary,
                                              weights_antimer,
                                              weight_error),
                         verts_new,
                         iteration_count=max_iterations,
                         normalize_func=normalize_func,
                         grad_tolerance=1e-4)"""
    
    #"""
    weights_dist = 0.1 * weights_water * weights_pop
    weights_area = 1 * weights_water * weights_pop
    weight_boundary = 1e-9
    weights_antimer *= 0.1
    weight_error = 0
    """verts_new = minimize(cost_grad_func_maker(weights_dist,
                                              weights_area,
                                              weight_boundary,
                                              weights_antimer,
                                              weight_error),
                         verts_new,
                         iteration_count=max_iterations,
                         normalize_func=normalize_func,
                         grad_tolerance=1e-4)"""

    weights_dist = 0.01 * weights_water * weights_pop
    weights_area = 1 * weights_water * weights_pop
    weight_boundary = 1e-9
    weights_antimer *= 0.1
    weight_error = 0
    verts_new = minimize(cost_grad_func_maker(weights_dist,
                                              weights_area,
                                              weight_boundary,
                                              weights_antimer,
                                              weight_error),
                         verts_new,
                         iteration_count=max_iterations,
                         normalize_func=normalize_func,
                         grad_tolerance=1e-7)
    #"""
    if sphere_first:
        set_up_plot(1.02, 1.02)
    else:
        set_up_plot(3.5, 2)
    mesh_final = Mesh(verts=verts_new, tris=mesh.tris)
    plot_mesh(mesh_final, np.clip(land_portions, 0, 1))
    poly_list, is_closed_list = poly_list_from_borders(WORLD_BORDERS_DATA_FLAT)
    poly_list, is_closed_list = interrupt_polygon_list_antimeridian(poly_list,
                                                            is_closed_list,
                                                            shift_degrees=11)
    poly_list_proj = polys_old_mesh_to_new(poly_list, mesh, mesh_final)
    plot_polygons(poly_list_proj, is_closed_list)
    return mesh_final


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
    verts_new = minimize(cost_grad_func_maker(0.2),
                         initial_state,
                         iteration_count=it_count,
                         #normalize_func=normalize_func)
                         grad_tolerance=1e-2,
                         )
    verts_new = minimize(cost_grad_func_maker(0.001),
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


def set_up_plot(xlim=1.02, ylim=1.02, y_offset=0):
    plt.cla()
    plt.gca().set_aspect("equal")
    plt.gca().set_xlim(-xlim, xlim)
    plt.gca().set_ylim(-ylim + y_offset, ylim + y_offset)
    plt.show()


def plot_mesh(mesh, tri_vals=None):
    if tri_vals is not None:
        assert mesh.tris.shape[0] == tri_vals.shape[0]
    for i, tri in enumerate(mesh.tris):
        a, b, c = mesh.verts[tri]
        if a.shape[0] == 3:
            if np.cross(b-a, c-a)[2] <= 0:
                continue
        a2d, b2d, c2d = a[0:2], b[0:2], c[0:2]
        xs, ys = np.column_stack([a2d, b2d, c2d, a2d])
        if tri_vals is not None:
            color = rgb2hex((0, tri_vals[i], 0))
            plt.fill(xs, ys, color)
        plt.plot(xs, ys, c="b", linewidth=0.2)


def plot_polygons(poly_list, is_closed_list):
    ax = plt.gca()
    for i, poly in enumerate(poly_list):
        poly_draw = poly[:, 0:2]
        ax.add_patch(Polygon(poly_draw,
                             closed=is_closed_list[i],
                             fill=False))


def main():
    plt.ion()


if __name__ == "__main__":
    main()
