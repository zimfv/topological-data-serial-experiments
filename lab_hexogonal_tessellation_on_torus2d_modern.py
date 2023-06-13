import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from package.filtredcells import FiltredCells
from package.tessellations import get_hexagonal_tessolation_on_torus2d



clock = time.perf_counter()

sizes = np.arange(2, 129)
experiments = 100


print('Calculating giant cycle births for {0} experiments for {1} different 3d-torus splits by cubes.'.format(experiments, len(sizes)))

bar0 = tqdm(total=len(sizes))
bar1 = tqdm(total=experiments)

filename = 'lab-results/lab_hexagonal_tessellation_on_torus2d_modern/size{0}_exp{1}.csv'

for size in sizes:
    bar0.set_postfix_str('size='+str(size))
    fc = get_hexagonal_tessolation_on_torus2d(n=size)
    filtrations = np.random.random([experiments, size**2])
    for i in range(experiments):
        filtration = filtrations[i]
        fc.set_filtration(filtration)
        fc.initialize_complex()
        #df_cycles = fc.get_giant_cycles()
        #df_cycles.to_csv(filename.format(size, i))
        df_info = fc.get_cycle_info()
        df_info.to_csv(filename.format(size, i))
        bar1.update()
        bar1.refresh()
    if  size != sizes[-1]:
        bar1.reset()
    bar0.update()
    bar0.refresh()

bar0.close()
bar1.close()
print('All sucessfuly calculated in {0: .4f} seconds.\n'.format(time.perf_counter() - clock))