import numpy as np
from itertools import product
from package.filtredcells import *
from package import complexes


def get_cubical_tessellation_on_torus2d(n, m=None, filtration=None):
    """
    Returns FiltredCells for cubic tessellation on 2-dimensionall torus.
    
    Parameters:
    -----------
    n, m: int
        Sizes of tessellation
        If m is None, that becomes same as n
    
    filtration : array length n*m or None
        Filtration values for cells. If that's None, that becames zeros.
        
    Returns:
    --------
    fc : FiltredCells
    """
    if m is None:
        m = n
    unit_simplices = np.array([[[0, 0], [0, 1], [1, 0]],
                               [[0, 1], [1, 0], [1, 1]]])
    verts = np.array(list(product(np.arange(n), np.arange(m))))
    cells = np.array([(v + unit_simplices[0], v + unit_simplices[1]) for v in verts]) 
    cells %= np.array([n, m])
    cells *= np.array([1, m])
    cells = cells.sum(axis=-1)
    unit = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    cords = np.array([v + unit for v in verts])
    if filtration is None:
        filtration = np.zeros(n*m)
    fc = FiltredCells(cells, filtration=filtration)
    fc.set_cords(cords)
    return fc


def get_hexagonal_tessolation_on_torus2d(n, m=None, d=0, filtration=None, undioganal=True):
    """
    Returns FiltredCells for hexagonal tessellation on 2-dimensionall torus.
    
    Parameters:
    -----------
    n, m: int
        Sizes of tessellation
        If m is None, that becomes same as n
    
    d : int
        Difference for gluing of bottom and top sides
    
    filtration : array length n*m or None
        Filtration values for cells. If that's None, that becames zeros.
    
    undiagonal : bool
        If that's True picture becomes more rectangular by changing cords 
    
    Returns:
    --------
    fc : FiltredCells
    """
    if m is None:
        m = n
    
    hexagons = []
    for j in range(m):
        for i in range(n):
            hexagon = [2*n*j + 2*i, 
                       2*n*j + 2*i + 1, 
                       2*n*j + (2*i + 2)%(2*n), 
                       2*n*(j+1) + (2*i + 1)%(2*n),
                       2*n*(j+1) + (2*i + 0)%(2*n),
                       2*n*(j+1) + (2*i - 1)%(2*n)]
            hexagons.append(hexagon)
    hexagons = np.array(hexagons)
    hexagons[hexagons >= 2*n*m] = (hexagons[hexagons >= 2*n*m] + 2*d) % (2*n)
    cells = np.array([(hexagon[[0, 1, 2]], 
                       hexagon[[2, 3, 4]], 
                       hexagon[[4, 5, 0]], 
                       hexagon[[0, 2, 4]]) for hexagon in hexagons])
    cords = []
    for j in range(m):
        for i in range(n):
            center = np.array([(j + 2*i), 1.5*j])
            if undioganal:
                center[0] %= 2*n
            cords.append(np.array([[ 1.0,  0.5], 
                                   [ 0.0,  1.0], 
                                   [-1.0,  0.5], 
                                   [-1.0, -0.5], 
                                   [ 0.0, -1.0], 
                                   [ 1.0, -0.5]]) + center)
    if filtration is None:
        filtration = np.zeros(n*m)
    fc = FiltredCells(cells, filtration=filtration)
    fc.set_cords(cords)
    return fc


def get_triangle_tessolation_on_torus2d(n, m=None, filtration=None, undioganal=True):
    """
    Returns FiltredCells for triangle tessellation on 2-dimensionall torus.
    
    Parameters:
    -----------
    n, m: int
        Sizes of tessellation
        If m is None, that becomes same as n
    
    filtration : array length n*m or None
        Filtration values for cells. If that's None, that becames zeros.
    
    undiagonal : bool
        If that's True picture becomes more rectangular by changing cords 
    
    Returns:
    --------
    fc : FiltredCells
    """
    if m is None:
        m = n
        
    indices = np.arange(n*m).reshape([n, m])
    cells = []
    cords = []
    for i in range(n):
        for j in range(m):
            cells.append([[indices[i, j], indices[(i+1)%n, j], indices[(i+1)%n, (j+1)%m]]])
            cells.append([[indices[i, j], indices[i, (j+1)%m], indices[(i+1)%n, (j+1)%m]]])
            cords.append([[i, j], [i + 1, j], [i + 1, j + 1]])
            cords.append([[i, j], [i, j + 1], [i + 1, j + 1]])
    cells = np.array(cells)
    cords = np.array(cords)
    
    if filtration is None:
        filtration = np.zeros(n*m)
    fc = FiltredCells(cells, filtration=filtration)
    fc.set_cords(cords)
    return fc


def get_cubical_tessellation_on_torus3d(n, m=None, k=None, filtration=None):
    """
    Returns FiltredCells for cubic tessellation on 3-dimensionall torus.
    
    Parameters:
    -----------
    n, m, k: int
        Sizes of tessellation
        If m is None, that becomes same as n
        If k is None, that becames same as m
    
    filtration : array length n*m*k or None
        Filtration values for cells. If that's None, that becames zeros.
        
    Returns:
    --------
    fc : FiltredCells
    """
    if m is None:
        m = n
    if k is None:
        k = m
    verts = np.array(list(product(np.arange(n), np.arange(m), np.arange(k))))
    unit_simplices = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], 
                               [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], 
                               [[0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]], 
                               [[1, 0, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0]], 
                               [[0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1]], 
                               [[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]]])
    cells = np.array([[v + s for s in unit_simplices] for v in verts]) 
    cells %= np.array([n, m, k])
    cells *= np.array([1, m, k*m])
    cells = cells.sum(axis=-1)
    if filtration is None:
        filtration = np.zeros(n*m*k)
    fc = FiltredCells(cells, filtration=filtration)
    return fc


def get_latticeD3_tessellation_on_torus3d(n: int, m=None, k=None, filtration=None):
    # That does not work correctly in situations when one of parameters n, m, k is less then 3.
    """
    Returns FiltredCells for tessellation of 3-dimensionall torus by voronoi cells of lattice D3.
    
    Parameters:
    -----------
    n, m, k: int
        Half sizes of tessellation
        If m is None, that becomes same as n
        If k is None, that becames same as m
    
    filtration : array length 4*n*m*k or None
        Filtration values for cells. If that's None, that becames zeros.
        
    Returns:
    --------
    fc : FiltredCells
    """
    if m is None:
        m = n
    if k is None:
        k = m
    verts = np.array(list(product(np.arange(2*n), np.arange(2*m), np.arange(2*k))))
    verts = verts[verts.sum(axis=1) % 2 == 0]
    if (np.array([n, m, k]) == 1).any():
        unit_cell = np.array([[[ 0,  0,  0], [ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]], 
                              [[ 0,  0,  0], [ 1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]], 
                              [[ 0,  0,  0], [ 1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]], 
                              [[ 0,  0,  0], [ 1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]], 
                              [[ 0,  0,  0], [-1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]], 
                              [[ 0,  0,  0], [-1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]], 
                              [[ 0,  0,  0], [-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]], 
                              [[ 0,  0,  0], [-1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]]])
        pass
    else:
        unit_cell = np.array([[[-1,  0,  0], [ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]], 
                              [[-1,  0,  0], [ 1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]], 
                              [[-1,  0,  0], [ 1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]], 
                              [[-1,  0,  0], [ 1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]]])
    arround = np.array([[[ 0.5,  0.5,  0.5], [ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]], 
                        [[ 0.5,  0.5, -0.5], [ 1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]], 
                        [[ 0.5, -0.5,  0.5], [ 1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]], 
                        [[ 0.5, -0.5, -0.5], [ 1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]], 
                        [[-0.5,  0.5,  0.5], [-1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]], 
                        [[-0.5,  0.5, -0.5], [-1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]], 
                        [[-0.5, -0.5,  0.5], [-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]], 
                        [[-0.5, -0.5, -0.5], [-1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]]])
    unit_cell = np.concatenate([unit_cell, arround])
    cells = [vert + unit_cell for vert in verts]
    cells = (2*np.array(cells)).astype(int)
    cells %= np.array([4*n, 4*m, 4*k])
    cells *= np.array([1, 4*n, 16*n*m])
    cells = cells.sum(axis=-1)
    
    if filtration is None:
        filtration = np.zeros(4*n*m*k)
    fc = FiltredCells(cells, filtration=filtration)
    
    # create cords
    if True:
        if m is None:
            m = n
        if k is None:
            k = m
        cords = []
        unit_cord = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        for vert in verts:
            center = np.array([vert[1], vert[2] + (2*k+2)*vert[0]])
            cord = center + unit_cord
            cords.append(cord)
        cords = np.array(cords)
        fc.set_cords(cords)
        # create yticks
        fc.yticks = np.arange(2*n*(2*k+2)) - 1
        fc.ylabels = np.array([(fc.yticks+1) % (2*k+2) - 1, (fc.yticks+1) // (2*k+2)]).transpose()
    
    return fc


def get_fcc_tessellation_on_torus3d(n: int, m=None, k=None, filtration=None):
    # new name for get_latticeD3_tessellation_on_torus3d
    return get_latticeD3_tessellation_on_torus3d(n, m, k, filtration)


def get_bcc_tessellation_on_torus3d(n, m=None, k=None, filtration=None, unit_code='34'):
    """
    Returns FiltredCells for cubic tessellation on 2-dimensionall torus.
    
    Parameters:
    -----------
    n, m, k: int
        Sizes of tessellation
        If m is None, that becomes same as n
        If k is None, that becomes same as 2*m
    
    filtration : array length n*m or None
        Filtration values for cells. If that's None, that becames zeros.
        
    unit_code : str
        Code of unit simplex complex - representation of truncated octahedron
        '24nr' : 24 simplices but there are not each face parallely moved to zero gives a reflection.
        '34' : 34 simlices
        '44' : 44 simlices
        '72' : 72 simlices
    
    Returns:
    --------
    fc : FiltredCells
    """
    if m is None:
        m = n
    if k is None:
        k = 2*m
    
    unit_simplices = {'24nr' : complexes.TruncatedOctahedron_24nr, 
                      '34' : complexes.TruncatedOctahedron_34, 
                      '44' : complexes.TruncatedOctahedron_44, 
                      '72' : complexes.TruncatedOctahedron_72}[str(unit_code)]
    basis = np.array([[2, 0, 0], [0, 2, 0], [1, 1, 1]])
    verts = np.array(list(product(np.arange(n), np.arange(m), np.arange(k))))
    verts = np.array([(vert.reshape([1, 3])@basis)[0] for vert in verts])
    verts %= np.array([2*n, 2*m, 2*k])
    verts
    #print(verts)
    cells = (2*np.array([vert + unit_simplices for vert in verts])).astype(int)
    cells %= np.array([4*n, 4*m, 2*k])
    cells *= np.array([1, 4*n, 16*n*m])
    cells = cells.sum(axis=-1)
    if filtration is None:
        filtration = np.zeros(n*m*k)
    fc = FiltredCells(cells, filtration=filtration)
    
    unit_cords = [np.array([(-0.5,  0.0, -1.0), ( 0.0, -0.5, -1.0), ( 0.5,  0.0, -1.0), ( 0.0,  0.5, -1.0)]), 
                  np.array([(-1.0,  0.0, -0.5), ( 0.0, -1.0, -0.5), ( 1.0,  0.0, -0.5), ( 0.0,  1.0, -0.5)]), 
                  np.array([(-1.0, -0.5,  0.0), (-1.0,  0.5,  0.0), (-0.5,  1.0,  0.0), ( 0.5,  1.0,  0.0), 
                            ( 1.0,  0.5,  0.0), ( 1.0, -0.5,  0.0), ( 0.5, -1.0,  0.0), (-0.5, -1.0,  0.0)]), 
                  np.array([(-1.0,  0.0,  0.5), ( 0.0, -1.0,  0.5), ( 1.0,  0.0,  0.5), ( 0.0,  1.0,  0.5)]), 
                  np.array([(-0.5,  0.0,  1.0), ( 0.0, -0.5,  1.0), ( 0.5,  0.0,  1.0), ( 0.0,  0.5,  1.0)])]
    cords = []
    for vert in verts:
        that_cords = []
        for unit_cord in unit_cords:
            cord = unit_cord + vert
            cord_z = (2*cord[:, 2]*(2*m + 2)) % (k*(2*m + 2))
            cord_z = np.array([np.zeros(len(cord_z)), cord_z]).transpose()
            cord = cord[:, :2] + cord_z
            that_cords.append(cord)
        cords.append(np.array(that_cords))
    
    fc.set_cords(cords)
    fc.yticks = np.unique(np.concatenate([np.concatenate(c) for c in cords])[:, 1])
    fc.yticks = np.arange(-1, k*(2*m + 2) - 1)
    fc.ylabels = [((y+1)//(2*m+2), y%(2*m+2)) for y in fc.yticks]
    return fc