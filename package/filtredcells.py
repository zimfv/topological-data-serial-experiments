import numpy as np
import gudhi as gd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from package.cycles import get_cycle_info


class FiltredCells:
    """
    Parameters:
    -----------
    vertices : array length nverts
        Names of vertices
    
    cords : list of matrices shape (nvertsincell, nspacedim) for each cell
        Coordinates of polygon vertices
    
    cells : array of lists of tuples length ncells
        List of simplices for each cell
    
    filtration : array length ncells
        Filtration value for each cell
        Default zeros.
    """
    def set_filtration(self, filtration=None):
        # 
        if filtration is None:
            self.filtration = np.zeros(len(self.cells))
        else:
            if len(filtration) != len(self.cells):
                raise ValueError('Wrong filtration')
            self.filtration = np.array(filtration)
    
    
    def __init__(self, cells, filtration=None):
        # 
        self.cells = np.array(cells, dtype=object)
        self.vertices = np.unique(np.concatenate(self.cells))
        self.set_filtration(filtration)
            
    
    def set_cords(self, cords):
        # Set cords parameter. Argument should be dict or matrix
        if len(cords) != len(self.cells):
            raise ValueError
        self.cords = cords
    
    
    def initialize_complex(self, filtration=None, dimension=None):
        # 
        if filtration is not None:
            self.set_filtration(filtration)
        self.simplextree = gd.SimplexTree()
        for i in range(len(self.cells)):
            cell = self.cells[i]
            value = self.filtration[i]
            for simplex in cell:
                self.simplextree.insert(simplex, filtration=value)
        if not dimension is None:
            self.simplextree.set_dimension(dimension)
        self.simplextree.compute_persistence()
        self.cycle_info = None
    
    
    def get_cycle_info(self):
        # 
        self.cycle_info = get_cycle_info(self.simplextree)
        return self.cycle_info
    
    
    def get_giant_cycles(self):
        # 
        if self.cycle_info is None:
            self.get_cycle_info()
        return self.cycle_info[self.cycle_info['Death'] == np.inf][['Birth', 'Dimension']]
    
    
    def persistent_euler(self):
        # 
        if self.cycle_info is None:
            self.get_cycle_info()
        cvals = np.unique(np.concatenate([self.cycle_info[['Birth', 'Death']].values, np.array([[np.inf, -np.inf]])]))
        df = pd.DataFrame({'A' : cvals[:-1], 'B' : cvals[1:]})
        df['c'] = 0.5*df['A'] + 0.5*df['B']
        df['Betty'] = [self.simplextree.persistent_betti_numbers(c, c) for c in df['c']]
        df['Euler Char'] = [((-1)**np.arange(len(b))*np.array(b)).sum() for b in df['Betty']]
        return df[['A', 'B', 'Euler Char']]
    
    def draw_filtration(self, colormap='winter', edgecolor='orangered', edgewidth=2, ax=plt):
        # 
        for i in range(len(self.cells)):
            color = cm[colormap]((self.filtration[i] - self.filtration.min())/(self.filtration.max() - self.filtration.min()))
            if len(self.cords[i].shape) == 2:
                x = self.cords[i][:, 0]
                y = self.cords[i][:, 1]
                ax.fill(x, y, color=color)
                ax.plot(np.append(x, x[0]), np.append(y, y[0]), color=edgecolor, linewidth=edgewidth)
            else:
                for cords in self.cords[i]:
                    x = cords[:, 0]
                    y = cords[:, 1]
                    ax.fill(x, y, color=color)
                    ax.plot(np.append(x, x[0]), np.append(y, y[0]), color=edgecolor, linewidth=edgewidth)