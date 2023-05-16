import numpy as np
import pandas as pd
from tqdm import tqdm

sizes = np.arange(3, 350)
experiments = 100
filename = 'lab-results/lab_cubical_tessellation_on_torus2d/size{0}_exp{1}.csv'

res = []
bar = tqdm(total=len(sizes))
for size in sizes:
    dfs = []
    for exp in range(experiments):
        dfs.append(pd.read_csv(filename.format(size, exp))[['Birth', 'Dimension']])
    if not np.all([list(df['Dimension']) == [0, 1, 1] for df in dfs]):
        print('ПИЗДЕЦ!\nThere are some noncorrect dataframes:')
        for df in dfs:
            if list(df['Dimension']) != [0, 1, 1]:
                print(df, '\n')
        break
    births01 = np.array([df['Birth'][0] for df in dfs])
    births11 = np.array([df['Birth'][1] for df in dfs])
    births12 = np.array([df['Birth'][2] for df in dfs])
    res.append({'Size' : size, 
                '1st 0-dim birth mean' : births01.mean(), 
                '1st 0-dim birth var' : births01.var(), 
                '1st 0-dim birth min' : births01.min(), 
                '1st 0-dim birth max' : births01.max(), 
                '1st 1-dim birth mean' : births11.mean(), 
                '1st 1-dim birth var' : births11.var(), 
                '1st 1-dim birth min' : births11.min(), 
                '1st 1-dim birth max' : births11.max(), 
                '2nd 1-dim birth mean' : births12.mean(), 
                '2nd 1-dim birth var' : births12.var(), 
                '2nd 1-dim birth min' : births12.min(), 
                '2nd 1-dim birth max' : births12.max()})
    bar.update()
bar.close()
res = pd.DataFrame(res)
res.to_csv('lab-results/lab_cubical_tessellation_on_torus2d_aggregated.csv', index=False)
print('File lab-results/lab_cubical_tessellation_on_torus2d_aggregated.csv sucessfuly created.')