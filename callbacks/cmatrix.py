
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('ggplot')
mpl.rc('font', size=15)
mpl.rc('figure', figsize=(8, 5))

import torch

from sklearn.metrics import confusion_matrix

import seaborn as sb


class CMatrix:
    '''Confusion matrix '''
    
    def __init__(self, labels='auto'):
        self.cmatrix = None
        self.labels = labels
    
    @property
    def cmatrix_norm(self):  
        if self.cmatrix is None:
            return None
        else:
            return self.cmatrix/self.cmatrix.sum(1)
    
    def update(self, y_pred, y_true):
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.data.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.data.cpu().numpy()
        
        cmatrix = confusion_matrix(y_true, y_pred)
        if self.cmatrix is None:
            self.cmatrix = cmatrix
        else:
            self.cmatrix += cmatrix
            
    def reset(self):
        self.cmatrix = None
        
    def plot(self, norm=False, cmap=plt.cm.Blues):
        ''' Plot confusion matrix and return figures'''
        
        cm = self.cmatrix_norm if norm else self.cmatrix
        
        # plot confusion matrix
        fig, ax = plt.subplots(1)
        sb.heatmap(cm, ax=ax, annot=True, fmt='.3g', cmap=cmap,
                   xticklabels=self.labels, yticklabels=self.labels)
        ax.set(xlabel='Predicted label', ylabel='True label')
        return fig, ax