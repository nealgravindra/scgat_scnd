
import sys
sys.path.append('/home/ngr4/project/scnd/scripts/')
import model as scgatmodels
import train as scgattrainer
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

def load_slim_pkl(slim_pkl):
    tic = time.time()
    with open(slim_pkl, 'rb') as f:
        data = pickle.load(f)
        f.close()
    print('data loaded in {:.0f}-s'.format(time.time() - tic))
    print('keys:', data.keys())
    print('')
    return data

def run(exp, target, trial, lr=0.001, min_nb_epochs=500, patience=100, scheduler=True):

    # load data
    slim_pkl = '/home/ngr4/project/scnd/data/processed/mouse_220808_model_data_slim.pkl'
    data = load_slim_pkl(slim_pkl)
    trainer = scgattrainer.trainer(
        data['pg_data'],
        data['metadata'],
        target,
        exp=exp,
        trial=trial,
        n_epochs=5000,
        min_nb_epochs=min_nb_epochs,
        lr=lr,
        weight_decay=5e-4,
        batch_size=32,
        scheduler=scheduler,
        patience=patience,
        model_savepath='/home/ngr4/scratch60/scnd_model_zoo',
        result_file='/home/ngr4/project/scnd/results/scgat_labelcrct.csv')
        
    
    # main
    trainer.fit()
    trainer.test()

    # save trainer
    filename = 'trainer_{}_{}_n{}.pkl'.format(exp, target, trial)
    filename = os.path.join(model_savepath, filename)
    
    # save space 
    out = trainer.log
    with open(filename, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close() 
        
    print('\n\nDONE with exp:{}\ttarget:{}\ttrial:{}'.format(exp, target, trial))
    print('  trainer dumped in {}'.format(filename))
    print('  ... EXITING.')


def longrun(exp, target, trial):
    run(exp, target, trial, lr=5e-4, min_nb_epochs=2000, patience=500, scheduler=False)

def scgat_scnd_v32(exp, target, trial):
    run(exp, target, trial, lr=0.001, min_nb_epochs=500, patience=100)

if __name__ == '__main__':
    
    # grab params
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, help="one of y_genotype_crct or y_genotime_crct")
    parser.add_argument('--target', type=str, help="one of y_genotype_crct or y_genotime_crct")
    parser.add_argument('--trial', type=int)

    args = parser.parse_args()

    exp = args.exp
    target = args.target
    trial = args.trial

    if exp == 'longrun':
        longrun(exp, target, trial)
    elif exp == 'scgatscndv32':
        scgat_scnd_v32(exp, target, trial)


