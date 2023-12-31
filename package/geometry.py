import itertools
import numpy as np


def collinear(a, b):
    # returns True if vectors a and b are collinear
    if len(a) != len(b):
        raise ValueError('Vectors has different dimensions')
    if ((a == 0) != (b == 0)).any():
        return False
    l = a[a != 0] / b[b!= 0]
    return (l[1:] == l[:-1]).all()


def sets_in_hyperspace(set0, set1, eps=10**-6):
    # returns True if sets are in 1 hyperspace
    g = True
    for vector in set1:
        matrix = np.array(set0)
        matrix -= vector
        # np.linalg.det can be not accurate, so I use eps instead zero
        if abs(np.linalg.det(matrix)) >= eps:
            g = False
            break
    return g


def get_hyperspaces(points, eps=10**-6):
    """
    Returns 2 arrays defining hyperspaces which is defined by some points from cloud points
    
    Parameters:
    -----------
    points : np.array dim 2
    
    eps: float
    
    Returns:
    matrices: array of matrices
        Each matrix is set of points, defining hyperspace
    
    indices: list of int lists
        Each list is set of point indices, lying in hyperspace
    """
    dim = points.shape[1]
    matrices = []
    for simplex in itertools.combinations(np.arange(len(points)), dim):
        matrix = points[np.array(simplex)]
        if np.linalg.matrix_rank(matrix) > dim-1:
            matrices.append(matrix)
    indices = [[] for matrix in matrices]
    for i in range(len(points)):
        point = points[i]
        for k in range(len(matrices)):
            matrix = matrices[k]
            if abs(np.linalg.det(matrix - point.reshape([1, dim]))) < eps:
                indices[k].append(i)
    vals = [str(i) for i in indices]
    vals, idx_start = np.unique(vals, return_index=True)
    matrices = np.array(matrices)[idx_start]
    indices = [indices[i] for i in idx_start]
    return matrices, indices


def get_hyperspaces_containing(points, p=0, eps=10**-6):
    """
    Returns 2 arrays defining hyperspaces containing point p and defined by some points from cloud points
    
    Parameters:
    -----------
    points : np.array dim 2
    
    p : float or float array
    
    eps: float
    
    Returns:
    matrices: array of matrices
        Each matrix is set of points, defining hyperspace
    
    indices: list of int lists
        Each list is set of point indices, lying in hyperspace
    """
    dim = points.shape[1]
    p = p*np.ones(dim)
    matrices = []
    for simplex in itertools.combinations(np.arange(len(points)), dim-1):
        matrix = np.concatenate([p.reshape([1, dim]), points[np.array(simplex)]])
        if np.linalg.matrix_rank(matrix) > dim-2:
            matrices.append(matrix)
    indices = [[] for matrix in matrices]
    for i in range(len(points)):
        point = points[i]
        for k in range(len(matrices)):
            matrix = matrices[k]
            if abs(np.linalg.det(matrix - point.reshape([1, dim]))) < eps:
                indices[k].append(i)
    vals = [str(i) for i in indices]
    vals, idx_start = np.unique(vals, return_index=True)
    matrices = np.array(matrices)[idx_start]
    indices = [indices[i] for i in idx_start]
    return matrices, indices


def get_max_hyperspace_containing(points, p=0, eps=10**-6):
    # returns matrix of points, defining hyperspace, containing p and max number of points from given
    m_list, h_list = get_hyperspaces_containing(points, p=p, eps=eps)
    lengths = np.array([len(hl) for hl in h_list])
    index = np.where(lengths == max(lengths))[0][0]
    return m_list[index]


def get_min_hyperspace_containing(points, p=0, eps=10**-6):
    # returns matrix of points, defining hyperspace, containing p and max number of points from given
    m_list, h_list = get_hyperspaces_containing(points, p=p, eps=eps)
    lengths = np.array([len(hl) for hl in h_list])
    index = np.where(lengths == min(lengths))[0][0]
    return m_list[index]


def get_hyperspace_value(points, p=0):
    # returns hyperspace value for point p for hyperspace defined by given points
    #"""
    dim = points.shape[1]
    matrix = points[:dim].copy()
    v = matrix[0].copy()
    matrix[0] = p
    matrix = matrix - v
    value = np.linalg.det(matrix)
    return value


def contains(spc, p=0, eps=10**-6):
    # returns True if hyperspace containing all points from spc also contains point p (which default is zero)
    return abs(get_hyperspace_value(spc, p)) < eps
    

def get_normal(points):
    # returns normal of d-dimensional hyperspace using d+1 given points dim d+1
    basis = points[1:] - points[0]
    normal = np.linalg.solve(np.concatenate([basis, np.random.random(size=[1, basis.shape[1]])]), np.append(np.zeros(len(basis)), 1))
    normal = normal / np.linalg.norm(normal)
    if np.linalg.det(points - normal) < 0:
        normal *= -1
    return normal


def reflect(points, vector):
    # returns reflection of points relative to a vector
    # s_{v} p = p - 2<p,v>/<v,v>*v
    coeff = 2 * np.dot(points, vector) / np.dot(vector, vector)
    new_points = points - coeff.reshape(len(coeff), 1)*vector.reshape([1, len(vector)])
    return new_points


def get_half(points, spc, eps=10**-6):
    # returns half of points which are in nonnegative side of hyperspaces defined by points from spc
    # half = np.array([np.linalg.det(spc - point) for point in points]) >= -eps
    half = np.array([get_hyperspace_value(spc, point) for point in points])
    half = points[half >= -eps]
    return half


def get_faces(points, r=None, with_matrices=False):
    # returns indexes of all faces
    dim = points.shape[1]
    faces = []
    matrices, indices = get_hyperspaces(points)
    status = np.zeros(len(matrices), dtype=bool)
    for i in range(len(status)):
        values = np.array([get_hyperspace_value(matrices[i], p) for p in points])
        if r is not None:
            values = np.round(values, r)
        if (values >= 0).all() or (values <= 0).all():
            status[i] = True
    faces = np.array(indices, dtype=object)[status]
    if with_matrices:
        return faces, np.array(matrices)[status]
    return faces


def get_opposite_faces(points, r=6):
    """
    Returns pairs of opposite faices
    
    Parameters:
    -----------
    points : np.array dim 2
    
    Returns:
    --------
    pairs_faces : list
        list of pairs of tuple indices
    
    pairs_normals : list
        list of normals arrays
    """
    faces, matrices = get_faces(points, r=r, with_matrices=True)
    normals = np.array([get_normal(matrix) for matrix in matrices])
    if r is not None:
        normals = np.round(normals, r)
    pairs_faces = []
    pairs_normals = []
    for i0, i1 in itertools.combinations(np.arange(len(normals)), 2):
        if collinear(normals[i0], normals[i1]):
            pairs_faces.append((faces[i0], faces[i1]))
            pairs_normals.append(normals[i0])
    return pairs_faces, pairs_normals