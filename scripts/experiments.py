
import sys
sys.path.append('/home/ngr4/project/scnd/scripts/')
import model as scgatmodels
import train as scgattrain
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

if __name__ == '__main__':
    
    # grab params
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, help="one of y_genotype_crct or y_genotime_crct")
    parser.add_argument('--trial', type=int)

    args = parser.parse_args()

    target = args.target
    trial = args.trial
    
    ####
    # exp name
    ####
    exp = 'scgatscndv2'
    model_savepath = '/home/ngr4/scatch60/scnd_model_zoo'
    ####

    # load data
    slim_pkl = '/home/ngr4/project/scnd/data/processed/mouse_220808_model_data_slim.pkl'

    # load data
    tic = time.time()
    with open(slim_pkl, 'rb') as f:
        data = pickle.load(f)
        f.close()
    print('data loaded in {:.0f}-s'.format(time.time() - tic))
    print('keys:', data.keys())
    print('')
    
    trainer = scgattrainer.trainer(
        data['pg_data'],
        data['metadata'],
        target,
        exp=exp,
        trial=trial,
        n_epochs=5000,
        min_nb_epochs=500,
        lr=0.001,
        weight_decay=5e-4,
        batch_size=32,
        patience=100,
        model_savepath=model_savepath,
        result_file='/home/ngr4/project/results/scgat_labelcrct.csv')
    trainer.fit()
    trainer.test()
    
    # save trainer
    filename = 'trainer_{}_{}_n{}.pkl'.format(exp, target, trial)
    filename = os.path.join(model_savepath, filename)
    with open(filename, 'wb') as f:
        pickle.dump(trainer, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close() 
        
    print('\n\nDONE with exp:{}\ttarget:{}\ttrial:{}'.format(exp, target, trial))
    print('  trainer dumped in {}'.format(filename))
    print('  ... EXITING.')