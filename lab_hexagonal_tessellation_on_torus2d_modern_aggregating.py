import numpy as np
import pandas as pd
from tqdm import tqdm

sizes = np.arange(2, 129)
experiments = np.arange(100)

input_filename = 'lab-results/lab_hexagonal_tessellation_on_torus2d_modern/size{0}_exp{1}.csv'
output_filename = 'lab-results/lab_hexagonal_tessellation_on_torus2d_modern_aggregated/size{0}.csv'

bar_sizes = tqdm(total=len(sizes), desc='Sizes')
bar_download = tqdm(total=len(experiments), desc='Downloading')

for size in sizes:
    dfs = []
    for exp in experiments:
        dfs.append(pd.read_csv(input_filename.format(size, exp))[['Birth', 'Death', 'Dimension']])
        dfs[-1]['Space'] = exp
        bar_download.update()
    
    df_res = pd.concat(tqdm(dfs, desc='Concat size: {0}'.format(size))) 
    df_res.to_csv(output_filename.format(size), index=False)
    if size != sizes[-1]:
        bar_download.reset()
    bar_sizes.update()