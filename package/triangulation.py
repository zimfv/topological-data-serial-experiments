import itertools
import numpy as np
from scipy.spatial import Delaunay
from package import geometry

from icecream import ic
ic.disable()

def triangulate_body(points):
    # returns simplices bt simple Delaunay triangulation
    half_simplex_index = np.array(Delaunay(points).simplices)
    half_simplex_vals = points[half_simplex_index]
    return half_simplex_vals


def reflect_simplices(simplices, vector):
    # returns simplices refected relative to a vector
    new_simplices = np.array([geometry.reflect(simplex, vector) for simplex in simplices])
    return new_simplices


def triangulate_body_reflecting_half(points, p=0, choose='max', eps=10**-6):
    # returns simplices, defined by triangulation of half pints and reflecting the result
    matrix = {'max': geometry.get_max_hyperspace_containing, 
              'min': geometry.get_min_hyperspace_containing}[choose](points, p=p, eps=eps)
    normal = geometry.get_normal(matrix)
    half = geometry.get_half(points, matrix, eps=eps)
    half_simplex_index = np.array(Delaunay(half).simplices)
    half_simplex_vals = half[half_simplex_index]
    refl_simplex_vals = reflect_simplices(half_simplex_vals, normal)
    simplex_vals = np.concatenate([half_simplex_vals, 
                                   refl_simplex_vals])
    return simplex_vals


def get_subcomplex(triangulation, matrix, r=6):
    """
    Returns subcomplex of given triangulation such that lies in hyperspace defined by matrix.
    
    Parameters:
    -----------
    triangulation : np.array shape (N, dim+1, dim)
        List os simplices in given complex
    
    matrix : np.array shape (dim, dim)
        points, defining hyperspace
        
    Returns:
    --------
    new_triangulation
    """
    dim = matrix.shape[1]
    new_triangulation = []
    for simplex in triangulation:
        status = np.array([geometry.contains(matrix, p) for p in simplex])
        new_simplex = simplex[status]
        if len(new_simplex) == dim:
            new_triangulation.append(new_simplex)
    return np.array(new_triangulation)


def equal_triangulations(triangulation_a, triangulation_b):
    # returns True if 2 triangulations are equal
    if triangulation_a.shape != triangulation_b.shape:
        return False
    triangulation_a = np.sort(triangulation_a, axis=2)
    triangulation_b = np.sort(triangulation_b, axis=2)
    triangulation_a = np.sort(triangulation_a, axis=1)
    triangulation_b = np.sort(triangulation_b, axis=1)
    triangulation_a = np.sort(triangulation_a, axis=0)
    triangulation_b = np.sort(triangulation_b, axis=0)
    return (triangulation_a == triangulation_b).all()


def is_symmetric(triangulation, r=6):
    # returns True if all opposite faces of triangulated polytop triangulated symmetrically
    dim = triangulation.shape[-1]
    points = np.concatenate(triangulation)
    points = np.unique(points, axis=0)
    pairs, normals = geometry.get_opposite_faces(points)
    ic(points)
    for i in range(len(pairs)):
        normal = normals[i]
        ic(pairs[i][0])
        face0 = points[np.array(pairs[i][0]).astype(int)]
        ic(face0)
        face1 = points[np.array(pairs[i][1]).astype(int)]
        matrix0 = face0[:dim]
        matrix1 = face1[:dim]
        for comb in itertools.combinations(np.arange(len(face0)), dim):
            if np.linalg.matrix_rank(face0[np.array(comb)]) == dim:
                matrix0 = points[np.array(comb)]
                break;
        for comb in itertools.combinations(np.arange(len(face1)), dim):
            if np.linalg.matrix_rank(face1[np.array(comb)]) == dim:
                matrix1 = points[np.array(comb)]
                break;
        ic(matrix0)
        ic(matrix1)
        triangulation0 = get_subcomplex(triangulation, matrix0, r=r)
        triangulation1 = get_subcomplex(triangulation, matrix1, r=r)
        if equal_triangulations(triangulation0, reflect_simplices(triangulation1, normal)):
            return True
    return False