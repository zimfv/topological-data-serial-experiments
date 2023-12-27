import numpy as np
from scipy.spatial import Delaunay
from package import geometry


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
    #print(f"matrix:\n{matrix}\nnormal: {normal}\nhalf:\n{half}")
    #print(f"half_simplex_vals:\n{half_simplex_vals}\nrefl_simplex_vals:\n{refl_simplex_vals}\n")
    simplex_vals = np.concatenate([half_simplex_vals, 
                                   refl_simplex_vals])
    return simplex_vals