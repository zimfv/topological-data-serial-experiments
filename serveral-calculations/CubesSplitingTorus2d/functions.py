import numpy as np
import pandas as pd
from gudhi import SimplexTree


def cube_simplex_generator(dimension):
    # Generates (dimension+1)-simplices 
    if dimension == 2:
        yield [np.array([0, 0]), 
               np.array([0, 1]), 
               np.array([1, 0])]
        yield [np.array([0, 1]), 
               np.array([1, 0]), 
               np.array([1, 1])]
    else:
        pass


def get_simplicies(matrix):
    # 
    dimension = len(matrix.shape)
    n = np.array(matrix.shape, dtype=int)
    coefs = np.zeros(dimension, dtype=int)
    for i in range(dimension):
        coefs[i] = n[:i].prod()
    
    cords = np.array(np.where(matrix)).transpose()
    simplicies = []
    for cord in cords:
        for vector in cube_simplex_generator(dimension):
            simplicies.append((cord + vector)%n)
    for i in range(len(simplicies)):
        simplicies[i] = (simplicies[i] * coefs).sum(axis=1)
    return simplicies


def get_torus_with_filtration(matrix):
    # 
    st = SimplexTree()
    ps = np.unique(np.concatenate([[0, 1], matrix.reshape(np.prod(matrix.shape))]))
    for p in ps:
        for simplex in get_simplicies(matrix <= p):
            st.insert(simplex, filtration=p)
    return st


def get_filtration_values(st: SimplexTree):
    # 
    filtration_values = np.unique([i[1] for i in st.get_filtration()])
    start_value = int(filtration_values.min())
    if start_value == filtration_values.min():
        start_value -= 1
    filtration_values = np.append(start_value, filtration_values)
    return filtration_values


def get_changing_values(st: SimplexTree):
    # 
    filtration_values = get_filtration_values(st)
    changing_values = [filtration_values[0]]
    for value in filtration_values:
        if st.persistent_betti_numbers(value, value) != st.persistent_betti_numbers(changing_values[-1], changing_values[-1]):
            changing_values.append(value)
    return np.array(changing_values)


def get_simplex_filtration_dict(st: SimplexTree):
    # returns dict, which keys are simplicies (tuples), and values are filtration values (floats)
    simplex_filtration = {}
    for i in st.get_filtration():
        simplex_filtration.update({tuple(i[0]) : i[1]})
    return simplex_filtration


def get_cycle_info(st: SimplexTree):
    # 
    info = pd.DataFrame(columns=['Birth', 'Death', 'Dimension'])
    
    changing_values = get_changing_values(st)
    eps = 0.5*(changing_values[1:] - changing_values[:-1]).min()
    simplex_filtration_dict = get_simplex_filtration_dict(st)
    
    for birth_simplex, death_simplex in st.persistence_pairs():
        birth = simplex_filtration_dict[tuple(np.sort(birth_simplex))]
        try: 
            death = simplex_filtration_dict[tuple(np.sort(death_simplex))]
        except KeyError:
            death = np.inf
            
        born = np.array(st.persistent_betti_numbers(birth + eps, -np.inf)) - np.array(st.persistent_betti_numbers(birth - eps, -np.inf))
        if (born != 0).sum() == 1:
            dimension = np.where(born != 0)[0][0]
        else:
            message = 'Я просто не придумал, что в этом случае делать. Но он маловероятен.'
            message += '\nОднако когда-то всё-таки случится. C чем я тебя и поздравляю!\n\n'
            message += 'А вообще, суть в том, что в один момент родилось несколько циклов разной размерности.'
            raise ValueError(message)
        
        info = info.append({'Birth' : birth,  
                            'Death' : death, 
                            'Dimension' : dimension
                           }, ignore_index=True)
    info = info.astype({'Birth': float, 'Death': float, 'Dimension': int})
    return pd.DataFrame(info)