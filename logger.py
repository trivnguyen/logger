
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

import torch
from tensorboardX import SummaryWriter


'''
    TensorBoard Data will be stored in './runs' path
'''

class Logger:

    def __init__(self, model_name, data_name):
        self.model_name = model_name
        self.data_name = data_name
        
        self.comment = '{}_{}'.format(model_name, data_name)
        self.data_subdir = '{}/{}'.format(model_name, data_name)

        # TensorBoard
        self.writer = SummaryWriter(comment=self.comment)
        
    def log_metric(self, train_metric, test_metric, name, epoch, n_batch, num_batches):
        
        if isinstance(train_metric, torch.Tensor):
            train_metric = train_metric.data.cpu().numpy()
        if isinstance(test_metric, torch.Tensor):
            test_metric = test_metric.data.cpu().numpy()

        step = Logger._step(epoch, n_batch, num_batches)
        self.writer.add_scalars('{}/{}'.format(self.comment, name), {
            'train': train_metric,
            'test': test_metric}, step)
        
    def display_status(self, epoch, num_epochs, n_batch, num_batches, 
                       train_metric, test_metric, name, show_epoch=True):
        
        # var_class = torch.Tensor
        if isinstance(train_metric, torch.Tensor):
            train_metric = train_metric.data.cpu().numpy()
        if isinstance(test_metric, torch.Tensor):
            test_metric = test_metric.data.cpu().numpy()
        
        if show_epoch:
            print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
                epoch,num_epochs, n_batch, num_batches)
                 )
        print('Train {0:}: {1:.4f}, Test {0:}: {2:.4f}'.format(
            name, train_metric, test_metric))

    def save_models(self, model, epoch, n_batch):
        out_dir = './data/models/{}'.format(self.data_subdir)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), '{}/epoch_{}_batch_{}'.format(out_dir, epoch, n_batch))
        
    def close(self):
        self.writer.close()
        
    # Private Functionality
    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch
    
    
class ClassifierLogger(Logger):
    
    def __init__(self, model_name, data_name):
        super().__init__(model_name, data_name)
        
    def save_binhist(self, binhist, epoch, n_batch):
        out_dir = './data/binhists/{}'.format(self.data_subdir)
        os.makedirs(out_dir, exist_ok=True) 
        
        fig, ax = binhist.plot()
        fig.savefig('{}/epoch_{}_batch_{}.png'.format(out_dir, epoch, n_batch))
        plt.close()
        
    def save_cmatrix(self, cmatrix, epoch, n_batch, cmap=plt.cm.Blues):
        out_dir = './data/cmatrices/{}'.format(self.data_subdir)
        os.makedirs(out_dir, exist_ok=True)

        # confusion matrix
        fig, ax = cmatrix.plot(norm=False, cmap=cmap)
        fig.savefig('{}/epoch_{}_batch_{}.png'.format(out_dir, epoch, n_batch))
        plt.close()
        
        # normalized confusion matrix
        fig, ax = cmatrix.plot(norm=True, cmap=cmap)
        fig.savefig('{}/norm_epoch_{}_batch_{}.png'.format(out_dir, epoch, n_batch))
        plt.close()
    