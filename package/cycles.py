import numpy as np
import pandas as pd
import gudhi as gd


def get_filtration_values(st: gd.SimplexTree):
    # 
    filtration_values = np.unique([i[1] for i in st.get_filtration()])
    start_value = int(filtration_values.min())
    if start_value == filtration_values.min():
        start_value -= 1
    filtration_values = np.append(start_value, filtration_values)
    return filtration_values


def get_changing_values(st: gd.SimplexTree):
    # 
    filtration_values = get_filtration_values(st)
    changing_values = [filtration_values[0]]
    for value in filtration_values:
        if st.persistent_betti_numbers(value, value) != st.persistent_betti_numbers(changing_values[-1], changing_values[-1]):
            changing_values.append(value)
    return np.array(changing_values)


def get_simplex_filtration_dict(st: gd.SimplexTree):
    # returns dict, which keys are simplicies (tuples), and values are filtration values (floats)
    simplex_filtration = {}
    for i in st.get_filtration():
        simplex_filtration.update({tuple(i[0]) : i[1]})
    return simplex_filtration



def get_cycle_info(st: gd.SimplexTree):
    # returns DataFrame which contains info about cycles: their birt and death values and dimensions
    info = pd.DataFrame(columns=['Birth', 'Death'])
    
    simplex_filtration_dict = get_simplex_filtration_dict(st)
    
    for birth_simplex, death_simplex in st.persistence_pairs():
        birth = simplex_filtration_dict[tuple(np.sort(birth_simplex))]
        try: 
            death = simplex_filtration_dict[tuple(np.sort(death_simplex))]
        except KeyError:
            death = np.inf
        dimension = len(birth_simplex) - 1
        info = pd.concat([info, pd.DataFrame([{'Birth' : birth, 'Death' : death, 'Dimension' : dimension}])], ignore_index=True)
    info = info.astype({'Birth': float, 'Death': float, 'Dimension': int})
    return pd.DataFrame(info)
    