
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('ggplot')
mpl.rc('font', size=15)
mpl.rc('figure', figsize=(8, 5))

import torch


class BinHist:
    '''Binary score histogram '''
    
    def __init__(self, bins=np.linspace(0., 1., 21), labels=None):
        self.score0 = np.zeros(len(bins)-1)
        self.score1 = np.zeros(len(bins)-1)
        self.bins = bins
        self.labels = labels

    def update(self, y_pred, y_true):
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.data.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.data.cpu().numpy()

        score0, _ = np.histogram(y_pred[(y_true == 0)], self.bins)
        score1, _ = np.histogram(y_pred[(y_true == 1)], self.bins)
        self.score0 += score0
        self.score1 += score1
            
    def reset(self):
        self.score0 = np.zeros(len(self.bins)-1)
        self.score1 = np.zeros(len(self.bins)-1)
        
    def plot(self, both=False):
        ''' Plot histogram '''
        
        fig, ax = plt.subplots(1)
        bc = .5*(self.bins[1:]+self.bins[:-1]) # bin center
        if not both:
            ax.hist(bc, self.bins, weights=self.score0, 
                    edgecolor='k', alpha=0.5, label=self.labels[0])
            ax.hist(bc, self.bins, weights=self.score1, 
                    edgecolor='k', alpha=0.5, label=self.labels[1])
        else:
            ax.hist(bc, self.bins, weights=self.score0+self.score1,
                    edgecolor='k', alpha=0.5, label='{} + {}'.format(*self.labels))
        ax.set(xlabel='Score', ylabel='Frequency')
        ax.legend()
        return fig, ax