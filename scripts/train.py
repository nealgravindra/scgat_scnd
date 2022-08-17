
import sys
sys.path.append('/home/ngr4/project/scnd/scripts/')
import model as scgatmodels
# import train as scgattrainer #todo: put trainer in seperate file
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
                 batch_size=32,
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
        self.pg_data = pg_data # may not want to store this
        self.metadata = metadata
        self.target_colname = target_colname
        self.exp = '{}_n{}'.format(exp, trial)
        self.n_epochs = n_epochs
        self.min_nb_epochs = min_nb_epochs
        self.patience = patience
        self.model_savepath = model_savepath
        self.result_file = result_path
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type=='cuda':
            torch.cuda.empty_cache()
        
        # initialize data
        ## modify y label
        y_train = torch.tensor(metadata.loc[pg_data['train'].y, target_colname].to_numpy(dtype=np.float32))
        y_val = torch.tensor(metadata.loc[pg_data['val'].y, target_colname].to_numpy(dtype=np.float32))
        y_test = torch.tensor(metadata.loc[pg_data['test'].y, target_colname].to_numpy(dtype=np.float32))
        #todo: modify pg_data objects with new label 
        #todo: add minibatchers per set as dataloader_train etc.
        
        # model
        self.n_features = pg_data['train'].x.shape[1]
        self.n_class = y_train.unique().shape[0]
        self.model = scgatmodels.scGAT_customforward(self.n_features, self.n_class)
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay) #todo: switch to adam and add LR scheduler

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
            output = self.model(batch.x)

            self.optimizer.zero_grad()
            loss = F.nll_loss(output, batch.y)
            loss.backward()
            self.optimizer.step()
            mb_loss.append(loss.item())
            mb_metric.append(accuracy(output, batch.y).item())
        
        # update loggers
        self.log['loss'].append(np.mean(mb_loss))
        self.log['acc'].append(np.mean(mb_metric))
        
    def val(self):
        mb_loss = []
        mb_metric = []
        
        self.model.eval()
        for i, batch in enumerate(self.dataloader_val):
            batch = batch.to(self.device)
            output = self.model(batch.x)
            loss_val = F.nll_loss(output, batch.y)
            mb_loss.append(loss_val.item())
            mb_metric.append(accuracy(output, batch.y).item())
        
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
            else:
                bad_counter += 1

            if bad_counter == patience:
                self.clear_modelpkls(best_epoch)
                self.model = self.model.load_state_dict(
                    torch.load(os.path.join(self.model_savepath, '{}-{}.pkl'.format(best_epoch, exp)), 
                                         map_location=self.device)) # torch.device('cpu') if loading full batch
                break
            elif epoch==(n_epochs-1):
                self.clear_modelpkls(best_epoch)
                torch.save(self.model.state_dict(), os.path.join(self.model_savepath, '{}-{}.pkl'.format(epoch,exp))) # also save last
#                 self.model = self.model.to(torch.device('cpu'))
                print('*NOTE*: {}-epoch patience not reached after {} epochs'.format(patience, n_epochs))

        print('\nOptimization finished!\tBest epoch: {}\tMax epoch: {}'.format(best_epoch, epoch))
        print('  exp: {}'.format(exp))
        print('  training time elapsed: {}-h:m:s\n'.format(str(datetime.timedelta(seconds=timer.sum()))))
        
        # save training details
        if self.result_file is not None:
            if os.path.exists(self.result_file):
                pd.DataFrame(self.log).to_csv(self.result_file, mode='a', header=False)
            else: 
                pd.DataFrame(self.log).to_csv(self.result_file)

        return self.log
        
    def test(self):
        '''

        NOTE: 
          - run after fit(). Model is on cpu
        '''
        # test 
#         device = torch.device('cpu')
#         self.model = self.model.to(self.device) # make sure model on gpu
        self.model.eval()
        for i, batch in enumerate(self.dataloader_test):
            batch = batch.to(self.device)
            output = self.model(batch)
            if i==0:
                self.y_test = batch.y
                self.yhat_test = output
            else:
                self.y_test = torch.cat((self.y_test, batch.y), dim=0)
                self.yhat_test = torch.cat((self.yhat_test, output), dim=0)
        loss_test = F.nll_loss(self.yhat_test, self.y_test).item()
        acc_test = accuracy(self.yhat_test, self.y_test).item()
        self.log['loss_test'] = loss_test
        self.log['acc_test'] = acc_test

        print('Test set eval:')
        print('  loss_test = {:.4e}\n  acc_test    = {:.4e}'.format(loss_test,acc_test))