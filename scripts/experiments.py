
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

if __name__ == '__main__':
    #todo: run experiments with only changing label and trial number
    
    # grab params
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--trial', type=int)

    args = parser.parse_args()

    dataset = args.dataset
    exp = args.exp
    trial = args.trial
    
    # exp format: model_task_dataset
    exp = '{}_{}'.format(dataset, exp)

    # load data
    if dataset=='hbec':
        data = scgatutils.load_datapkl('/home/ngr4/project/scgat/data/processed/hbec_scv2.pkl')
    elif dataset=='rvcse':
        data = scgatutils.load_datapkl('/home/ngr4/project/scgat/data/processed/rvcse_ycondition.pkl')
    elif dataset=='pseq':
        data = scgatutils.load_datapkl('/home/ngr4/project/scgat/data/processed/cwperturb_scv2.pkl')
    else:
        print('{} dataset not pre-processed. do sc.AnnData --> datapkl first.'.format(dataset))
        print('... exiting.')
        exit()
    
    # init model
    model = scgatmodelsdev.scGAT(data['pg_data']['train'].x.shape[1], data['pg_data']['train'].y.unique().shape[0])


    # get mini batches
    cluster_loader_dict = scgatdata.do_clustergcn(data['pg_data'])
    
    # train and evaluate
    results_df = train(cluster_loader_dict, model,
                       n_epochs=10000,
                       model_savepath='/home/ngr4/project/scgat/model_zoo/',
                       result_file='/home/ngr4/project/scgat/results/scgat_v3.csv')
