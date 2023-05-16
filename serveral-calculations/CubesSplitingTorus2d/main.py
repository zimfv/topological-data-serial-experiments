import pickle
from functions import *
from tqdm import tqdm
import time


with open('data.pkl', 'rb') as fp:
    data = pickle.load(fp)

N = data['experiments']
shapes = data['shapes']

print('Calculating giant cycle births for {0} experiments for {1} different torus splits by cubics.'.format(N, len(shapes)))

bar0 = tqdm(total=len(shapes))
bar1 = tqdm(total=N)
shape_giant_births = []
for j in range(len(shapes)):
    shape = shapes[j]
    bar0.set_postfix_str(str(shape))
    matrices = np.random.random([N, shape[0], shape[1]])
    giant_births = []
    for i in range(N):
        matrix = matrices[i]
        st = get_torus_with_filtration(matrix)
        st.compute_persistence()
        df = get_cycle_info(st)
        giant_births.append(df[(df['Death'] == np.inf)&(df['Dimension'] == 1)]['Birth'].values)
        bar1.update()
        bar1.refresh()
    if  j != len(shapes)-1:
        bar1.reset()
    bar0.update()
    bar0.refresh()
    giant_births = np.array(giant_births)
    shape_giant_births.append({'shape' : shape, 
                               'births 1st' : giant_births[:, 0], 
                               'births 2nd' : giant_births[:, 1]})
bar0.close()
bar1.close()

output_filename = 'giant_births.pkl'
with open(output_filename, 'wb') as fp:
    pickle.dump(shape_giant_births, fp)
    
print('Giant cycle births for {0} experiments for {1} different torus splits by cubics has been calculated and was written to file '.format(N, len(shapes)) + output_filename)