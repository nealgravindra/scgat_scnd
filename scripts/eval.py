import sys
sys.path.append('/home/ngr4/project/scgat/scripts/')
import utils as scgatutils
# import models as scgatmodels
import models_dev as scgatmodels
import load_data as scgatdata
import glob
import seaborn as sns
import matplotlib.pyplot as plt

import scanpy as sc
import pandas as pd
import os
import numpy as np
import warnings
import random
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from captum.attr import IntegratedGradients, NoiseTunnel

if __name__=='__main__':
    '''
    TODO (ngr):
      (1) make into modules, clearly customizable per data set
      
    '''
    # grab params
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()
    dataset = args.dataset
    
    if dataset=='hbec':
        datapkl = '/home/ngr4/project/scgat/data/processed/hbec_scv2.pkl'
        modelpkl = '/home/ngr4/project/scgat/model_zoo/9926-hbec_basic_yinftime2.pkl'
        target = None
    elif dataset=='rvcse':
        datapkl = '/home/ngr4/project/scgat/data/processed/rvcse_ycondition.pkl'
        modelpkl = '/home/ngr4/project/scgat/model_zoo/9962-rvcse_basic_ycond5.pkl'
        target = 3
    elif dataset=='pseq':
        datapkl = '/home/ngr4/project/scgat/data/processed/cwperturb_scv2.pkl'
        modelpkl = '/home/ngr4/project/scgat/model_zoo/1325-pseq_basic_yinf4.pkl'
        target = 2
    out_file = '/home/ngr4/project/scgat/data/processed/{}_yprime.csv'.format(dataset)
    out_file_null = '/home/ngr4/project/scgat/data/processed/{}_yprime_null.csv'.format(dataset)
    
    print('\nLoaded {} data set\n'.format(dataset))
    
    ################################################################################
    # load data and model
    ################################################################################
    data = scgatutils.load_datapkl(datapkl)

    # add idx 
    data['pg_data']['test'].y = torch.cat((data['pg_data']['test'].y.reshape(-1, 1), torch.arange(data['pg_data']['test'].y.shape[0]).reshape(-1, 1)), dim=1) 

    chk = False
    if chk:
        # check matching
        adata = data['adatas']['test']
        mismatch = [False if adata.obs.iloc[i, -1]==data['pg_data']['test'].y[i, 0].item() else True for i in range(adata.shape[0])]
        print('Any mis-match?: {}'.format(any(mismatch)))

    # md of interest
    adata = data['adatas']['test']
    md = adata.obs
    md['n_idx'] = list(range(md.shape[0]))

    # get mini batches
    cluster_loader_dict = scgatdata.do_clustergcn(data['pg_data'])

    # load model in Cpatum-friendly format
    device = torch.device('cpu')
    model = scgatmodels.scGAT_customforward(data['pg_data']['train'].x.shape[1], data['pg_data']['train'].y.unique().shape[0]).to(device)
    model.load_state_dict(torch.load(modelpkl, map_location=device)) 
    
    std = (0.05 * data['pg_data']['test'].x.mean()).item() # add std ~ 5% noise to each input
    
    # clear memory
    del data, adata

    # test 
    model.eval()
    for i, batch in enumerate(cluster_loader_dict['test']):
        batch = batch.to(device)
        output = model(batch.x, batch.edge_index)
        if i==0:
            y_test = batch.y
            yhat_test = output
        else:
            y_test = torch.cat((y_test, batch.y), dim=0)
            yhat_test = torch.cat((yhat_test, output), dim=0)
    loss_test = F.nll_loss(yhat_test, y_test[:, 0]).item()
    acc_test = scgatutils.accuracy(yhat_test, y_test[:, 0]).item()

    print('\nTest set eval:')
    print('  <loss_test>= {:.4f}\n  acc_test   = {:.4f}'.format(loss_test,acc_test))
    
    _, predicted_label = torch.topk(yhat_test, 1)
    predicted_label.squeeze_()
    shuffled_pred_labs = predicted_label[torch.randperm(predicted_label.shape[0])]
    
    # add output and predicted label to metadata
    pred_prob, predicted_label = torch.topk(yhat_test, 1)
    pred_prob = pred_prob.exp()
    predicted_label.squeeze_()

    md = md.reset_index()
    md.loc[y_test[:, 1].numpy(), 'predicted_label'] = predicted_label.numpy()
    md.loc[y_test[:, 1].numpy(), 'pred_prob'] = pred_prob.detach().numpy()
    for i in range(yhat_test.shape[1]):
        md.loc[y_test[:, 1].numpy(), 'output_{}'.format(i)] = yhat_test[:, i].detach().numpy()

    ################################################################################
    # get Yprime
    ################################################################################

    ## custom forward (does not work with utils.reshape fx)
    def custom_forward(x, pg_object):
        edge_index = pg_object.edge_index
    #     edge_attr =  utils.edge_set_reshape(pg_object).float()
        return model(x, edge_index)

    ig = IntegratedGradients(custom_forward)
    nt = NoiseTunnel(ig)

    results = pd.DataFrame()
    timer = scgatutils.timer()
    for i, batch in enumerate(cluster_loader_dict['test']):
        print('  ',i,':',batch)
        timer.start()
        
        # use true target if None specified
        if target is not None:
            target_temp = target
        else: 
            target_temp = batch.y[:, 0]
            
        attr = nt.attribute(batch.x, 
                            stdevs=std,
                            target=target_temp,
                            additional_forward_args=(batch,),
                            nt_type='smoothgrad', 
                            nt_samples=6)
        
        dt = pd.DataFrame(attr.numpy())
        del attr
        dt['md_idx'] = batch.y[:, 1].numpy()
        dt['y'] = batch.y[:, 0].numpy() 
        dt = dt.merge(md, left_on='md_idx', right_on='n_idx')
        results = results.append(dt, ignore_index=True)
        del dt
        results.to_csv(out_file)
        timer.stop()
        print('... through {} batches in {:.0f}-min'.format(i+1, (timer.sum())/60))
    print('\n...   finished.')
    
    ## null distribution 
    print('\nConstructing null set...')
    results = pd.DataFrame()
    timer = scgatutils.timer()
    for i, batch in enumerate(cluster_loader_dict['test']):
        print('  ',i,':',batch)
        timer.start()

        attr = nt.attribute(batch.x, 
                            stdevs=std,
                            target=predicted_label[torch.randperm(predicted_label.shape[0])][:batch.y.shape[0]],
                            additional_forward_args=(batch,),
                            nt_type='smoothgrad', 
                            nt_samples=6)

        dt = pd.DataFrame(attr.numpy())
        del attr
        dt['md_idx'] = batch.y[:, 1].numpy()
        dt['y'] = batch.y[:, 0].numpy() 
        dt = dt.merge(md, left_on='md_idx', right_on='n_idx')
        results = results.append(dt, ignore_index=True)
        del dt
        results.to_csv(out_file_null)
        timer.stop()
        print('... through {} batches in {:.0f}-min'.format(i+1, (timer.sum())/60))
    print('\n...   finished.')
    
