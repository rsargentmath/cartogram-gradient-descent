import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


PHI = (1 + np.sqrt(5)) / 2
OCTAHEDRON = (np.array([[1, 0, 0], [0, 1, 0],
                        [0, 0, 1], [-1, 0, 0],
                        [0, -1, 0], [0, 0, -1]]),
              np.array([[0, 1, 2], [0, 2, 4],
                        [0, 4, 5], [0, 5, 1],
                        [1, 3, 2], [1, 5, 3],
                        [2, 3, 4], [3, 5, 4]])
              )
ICOSAHEDRON = (np.array([[PHI, 1, 0], [-PHI, 1, 0],
                         [-PHI, -1, 0], [PHI, -1, 0],
                         [0, PHI, 1], [0, -PHI, 1],
                         [0, -PHI, -1], [0, PHI, -1],
                         [1, 0, PHI], [1, 0, -PHI],
                         [-1, 0, -PHI], [-1, 0, PHI]]) / np.sqrt(PHI + 2),
               np.array([[0, 3, 9], [0, 4, 8],
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
CUBE = (np.array([[1, 0, 0], [0, 1, 0],
                  [0, 0, 1], [-1, 0, 0],
                  [0, -1, 0], [0, 0, -1],
                  [a, a, a], [-a, a, a],
                  [a, -a, a], [-a, -a, a],
                  [a, a, -a], [-a, a, -a],
                  [a, -a, -a], [-a, -a, -a]]),
        np.array([[0, 6, 8], [0, 8, 12],
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
    return np.array(verts), np.array(tris)


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
    return verts_nzd, tris


# Mesh MUST be symmetric when reflecting across edges, otherwise
# vertices on edges may not line up.
def subdivide_mesh_sphere(mesh, n):
    tolerance = 1e-12
    # Subdivide each mesh triangle. Store new vertices and triangles.
    verts_og, tris_og = mesh
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
            if vec_norm(verts[ix] - verts[seen_ix]) < tolerance:
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
    return verts_final, tris_final


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
        initial_state, *,
        normalize_func=lambda x: x,
        learning_rate=0.05,
        iteration_count=100):
    state = initial_state.copy()
    for i in range(iteration_count):
        state -= learning_rate * grad_cost_func(state, i)
        state = normalize_func(state)
    return state


def plot_mesh(mesh):
    plt.figure()
    verts, tris = mesh
    for tri in tris:
        a, b, c = verts[tri]
        if a.shape[0] == 3:
            if np.cross(b-a, c-a)[2] <= 0:
                continue
        a2d, b2d, c2d = a[0:2], b[0:2], c[0:2]
        xs, ys = np.column_stack([a2d, b2d, c2d, a2d])
        plt.plot(xs, ys)
    plt.gca().set_aspect("equal")
    plt.show()
