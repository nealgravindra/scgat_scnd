
import sys
sys.path.append('/home/ngr4/project/scnd/scripts/')
import model as scgatmodels
import glob


import scanpy as sc
import pandas as pd
import os
import time
import datetime
import numpy as np
import warnings
import random
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import ClusterData, ClusterLoader

class timer(): 
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tic = time.time()

    def stop(self):
        self.times.append(time.time() - self.tic)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
    
# training evaluation metrics
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class trainer(object):
    def __init__(self, 
                 pg_data,
                 metadata,
                 target_colname,
                 exp='gtype_crct',
                 trial=0,
                 n_epochs=5000,
                 min_nb_epochs=500,
                 lr=0.001,
                 weight_decay=5e-4,
                 batch_size=32, # pretty much ignored, or represents roughly the number of sub-graphs
                 patience=100,
                 model_savepath='/home/ngr4/scatch60/scnd_model_zoo',
                 result_file='/home/ngr4/project/results/scgat_labelcrct.csv'):
        """Train, evaluate, and store model on single-cell data.
        
        Arguments:
          pg_data (dict): with keys=['train', 'val', 'test'] and Data() class from
            PyTorch Geometric with x, y as idx
          metadata (pd.DataFrame): with label information
          target_colname (str): specify colname in metadata that should be used for prediction
        """
        self.metadata = metadata
        self.target_colname = target_colname
        self.exp = '{}_{}_n{}'.format(exp, target_colname, trial)
        self.n_epochs = n_epochs
        self.min_nb_epochs = min_nb_epochs
        self.patience = patience
        self.model_savepath = model_savepath
        self.result_file = result_file
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type=='cuda':
            torch.cuda.empty_cache()
        
        # initialize data
        ## modify y label
        pg_data['train'].target = torch.LongTensor(metadata.loc[pg_data['train'].y, target_colname].to_numpy(dtype=np.float32))
        pg_data['val'].target = torch.LongTensor(metadata.loc[pg_data['val'].y, target_colname].to_numpy(dtype=np.float32))
        pg_data['test'].target = torch.LongTensor(metadata.loc[pg_data['test'].y, target_colname].to_numpy(dtype=np.float32))
        self.train_idx = {i:k for i, k in enumerate(pg_data['train'].y)}
        self.val_idx = {i:k for i, k in enumerate(pg_data['val'].y)}
        self.test_idx = {i:k for i, k in enumerate(pg_data['test'].y)}
        del pg_data['train'].y, pg_data['val'].y, pg_data['test'].y
        pg_data['train'].idx = torch.arange(pg_data['train'].x.shape[0])
        pg_data['val'].idx = torch.arange(pg_data['val'].x.shape[0])
        pg_data['test'].idx = torch.arange(pg_data['test'].x.shape[0])
        
#         self.pg_data['train'].y = np.array(pg_data['train'].y)
#         self.pg_data['val'].y = np.array(pg_data['val'].y)
#         self.pg_data['test'].y = np.array(pg_data['test'].y)
        
        # minibatcher
        self.dataloader_train = self.minibatcher(pg_data['train'], batch_size=batch_size)
        self.dataloader_val = self.minibatcher(pg_data['val'], batch_size=batch_size)    
        self.dataloader_test = self.minibatcher(pg_data['test'], batch_size=batch_size)
                
        # model
        self.n_features = pg_data['train'].x.shape[1]
        self.n_class = pg_data['train'].target.unique().shape[0]
        self.model = scgatmodels.scGAT_customforward(self.n_features, self.n_class)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adagrad(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay) 
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        # log 
        self.timer = timer()
        self.log = {
            'exp': self.exp,
            'hyperparams': {
                'patience': patience, 
                'lr': lr,
                'weight_decay': weight_decay,
                'batch_size': batch_size,
            },
            'loss': [], # averaged over minibatches
            'loss_val': [],
            'acc': [],
            'acc_val': [],
        }

        del pg_data # for safe measure
        
    def minibatcher(self, d, batch_size=None):
        if batch_size is None:
            batch_size = int((np.sqrt(d.x.shape[0]))/32) # default to 32
        cd = ClusterData(d,num_parts=int(np.sqrt(d.x.shape[0])))
        return ClusterLoader(cd,batch_size=batch_size, shuffle=True)
        
    def clear_modelpkls(self, best_epoch):
        files = glob.glob(os.path.join(self.model_savepath, '*-{}.pkl'.format(self.exp)))
        for file in files:
            epoch_nb = int(os.path.split(file)[1].split('-{}.pkl'.format(self.exp))[0])
            if epoch_nb != best_epoch:
                os.remove(file)
        
    def train(self):
        # log 
        mb_loss = []
        mb_metric = []
        
        # train
        self.model.train()
        for i, batch in enumerate(self.dataloader_train):
            batch = batch.to(self.device)
            output = self.model(batch.x, batch.edge_index)

            self.optimizer.zero_grad()
            loss = F.nll_loss(output, batch.target)
            loss.backward()
            self.optimizer.step()
            mb_loss.append(loss.item())
            mb_metric.append(accuracy(output, batch.target).item())
        
        # update loggers
        self.log['loss'].append(np.mean(mb_loss))
        self.log['acc'].append(np.mean(mb_metric))
        
    def val(self):
        mb_loss = []
        mb_metric = []
        
        self.model.eval()
        for i, batch in enumerate(self.dataloader_val):
            batch = batch.to(self.device)
            output = self.model(batch.x, batch.edge_index)
            loss_val = F.nll_loss(output, batch.target)
            mb_loss.append(loss_val.item())
            mb_metric.append(accuracy(output, batch.target).item())
        
        # update loggers
        self.log['loss_val'].append(np.mean(mb_loss))
        self.log['acc_val'].append(np.mean(mb_metric))
        
        return loss_val.item()
                
    def fit(self):

        bad_counter = 0
        best = np.inf 
        best_epoch = 0

        for epoch in range(self.n_epochs): 
            self.timer.start()
            self.train()
            loss_val = self.val()
            self.scheduler.step(loss_val)
            print('epoch:{}\tloss:{:.4e}\tacc:{:.4e}\tloss_val:{:.4e}\tacc_val:{:.4e}\tin:{:.2f}-s'.format(
                epoch,
                self.log['loss'][-1],
                self.log['acc'][-1],
                self.log['loss_val'][-1],
                self.log['acc_val'][-1],
                self.timer.stop()
            ))

            # save to model_zoo
            if loss_val < best and epoch+1 > self.min_nb_epochs:
                best = loss_val
                best_epoch = epoch
                self.clear_modelpkls(best_epoch)
                torch.save(self.model.state_dict(), 
                           os.path.join(self.model_savepath, '{}-{}.pkl'.format(epoch, self.exp)))
                bad_counter = 0
            elif epoch+1 > self.min_nb_epochs:
                bad_counter += 1

            if bad_counter == self.patience:
                self.clear_modelpkls(best_epoch)
                self.model.load_state_dict(
                    torch.load(os.path.join(self.model_savepath, '{}-{}.pkl'.format(best_epoch, self.exp)), 
                    map_location=self.device.type)) # torch.device('cpu') if loading full batch
                break
            elif epoch==(self.n_epochs-1):
                self.clear_modelpkls(best_epoch)
                torch.save(self.model.state_dict(), os.path.join(self.model_savepath, '{}-{}.pkl'.format(epoch,self.exp))) # also save last
#                 self.model = self.model.to(torch.device('cpu'))
                print('*NOTE*: {}-epoch patience not reached after {} epochs'.format(self.patience, self.n_epochs))

        print('\nOptimization finished!\tBest epoch: {}\tMax epoch: {}'.format(best_epoch, epoch))
        print('  exp: {}'.format(self.exp))
        print('  training time elapsed: {}-h:m:s\n'.format(str(datetime.timedelta(seconds=self.timer.sum()))))
        
        # save training details
        if self.result_file is not None:
            df = {}
            for k in self.log.keys():
                if isinstance(self.log[k], dict):
                    for kk in self.log[k].keys():
                        df['{}_{}'.format(k, kk)] = self.log[k][kk]
                else:
                    df[k] = self.log[k]
            if os.path.exists(self.result_file):
                pd.DataFrame(df).to_csv(self.result_file, mode='a', header=False)
            else: 
                pd.DataFrame(df).to_csv(self.result_file)

        
    def test(self, device=torch.device('cpu')):
        '''

        NOTE: 
          - run after fit(). Model is on cpu
        '''
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        # test 
        self.model = self.model.to(device) # make sure model on gpu
        self.model.eval()
        y_test = torch.empty(len(self.test_idx.keys()), )
        yhat_test = torch.empty(len(self.test_idx.keys()), self.n_class)
        count = 0
        for i, batch in enumerate(self.dataloader_test):
            n = batch.x.shape[0]
            batch = batch.to(device)
            output = self.model(batch.x, batch.edge_index)
            y, yhat = batch.target, output.detach()
            del batch, output
            yhat_test[count:count+n, :] = yhat
            y_test[count:count+n] = y
            del yhat, y
            count += n
        loss_test = F.nll_loss(yhat_test, y_test).item()
        acc_test = accuracy(yhat_test, y_test).item()
        self.log['loss_test'] = loss_test
        self.log['acc_test'] = acc_test

        print('Test set eval:')
        print('  loss_test = {:.4e}\n  acc_test    = {:.4e}'.format(loss_test,acc_test))
