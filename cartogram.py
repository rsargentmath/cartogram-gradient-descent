import numpy as np


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
    return np.array(verts), np.array(tris)


def vec_norm(v):
    return np.sqrt(v @ v)


def nzd(v):
    return v / vec_norm(v)


def tri_area_plane(a, b, c):
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
    return verts_nzd, tris


def this_tri_abc(verts, tri):
    return verts[tri[0]], verts[tri[1]], verts[tri[2]]


def tangent_space_matrix(a, b, c, clamp_to_sphere=False):
    dim = a.shape[0]
    assert dim == 2 or dim == 3
    if dim == 2:
        return np.identity(2)
    
    tolerance = 1e-12
    assert vec_norm(np.cross(b-a, c-a)) >= tolerance
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
        cost_func,
        grad_cost_func,
        initial_state,
        normalize_func=lambda x: x,
        learning_rate=0.05,
        iteration_count=100):
    state = initial_state.copy()
    for i in range(iteration_count):
        state -= learning_rate * grad_cost_func(state, i)
        state = normalize_func(state)
    return state


OCTAHEDRON_TRI = (np.array([1, 0, 0]),
                  np.array([0, 1, 0]),
                  np.array([0, 0, 1]))
phi = (1 + np.sqrt(5)) / 2
ICOSAHEDRON_TRI = (nzd(np.array([1, 0, phi])),
                   nzd(np.array([0, phi, 1])),
                   nzd(np.array([-1, 0, phi])))
