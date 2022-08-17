
import datetime
import sys
import pickle
from scipy import sparse
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns

import scanpy as sc
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
import random
from scipy import sparse

import torch
import torch.nn as nn
import torch.functional as F

from torch_geometric.data import Data, ClusterData, ClusterLoader

sc.settings.verbosity=2



# data
## data loader 
class pp_singlecell_data():
    # from adata (scanpy.AnnData obj)
    def __init__(self, adata_filename, label_colname, pkl_out=None, adata_out=None, 
                 return_joint_adata=False, return_adatas=True, return_pgdata=True):
        '''Load single-cell data, add splits, and prep for torch loading
        
        TODO:
          1. check normalization 
          2. add options for similar pre-process by pointing to cellranger
          3. if load from datapkl, don't go through slow pp steps
        
        Arugments:
          adata_filename (str): full filename to sc.AnnData object
          label_colname (str or list): column name in adata.obs slot to turn into class labels. If 
            label_colname exists, keep int class labels. Last label colname in list will get used as target 
          adata_out (str) : (optional, Default=None) specify the filepath to write modified adata to this filename
          return_joint_adata (bool): (optional, Default=False) if you want just the full adata back for plotting, etc.
          return_* (bool): (optional, Default=True) specify whether you want the data to return these things. E.g, if you're only 
            interested in plotting adata things, you don't need to create pytorch-geometric objects. And if you're only running models
            you don't need to store all the adata junk
          pkl_out (str): (optional, Default=None) if not none, filepath to dump all pp'ed data objs. Overwirtes return options
          
        '''
        self.adata_filename = adata_filename
        self.label_colname = label_colname
        self.pkl_out = pkl_out
        self.adata_out = adata_out
        self.return_joint_adata = return_joint_adata
        self.return_adatas = return_adatas
        self.return_pgdata = return_pgdata
        
        tic = time.time()
        
        print('Loading single-cell data for modeling')
        adata = sc.read(adata_filename)
        
        if True:
            # clean adata to minimize pkl size
            del adata.obsm
            del adata.layers
            del adata.uns
            del adata.obsp
        
        print('  ... checking for split')
        if 'split' not in adata.obs.columns.to_list() : # otherwise, assume splits are good
            print('        adding 70/15/15 train/test/val split')
            if adata_out is None:
                warnings.warn('Splits added to adata.obs. For reproducibility, save adata by specifying adata_out=filename_out.', UserWarning)
            adata = self.add_splits(adata, p_val=0.15, p_test=0.15)
        
        print('  ... checking for {} presence in adata.obs'.format(label_colname))
        # encode label via alphabetization for eventual torch --> LongTensor
        if isinstance(self.label_colname, list):
            for label_ in self.label_colname:
                adata = self.label_encoding(adata, label_colname=label_)
        else:
            adata = self.label_encoding(adata, label_colname=self.label_colname)
        
        if self.return_joint_adata or self.pkl_out is not None:
            if False:
                # recalculate the neighbor graph?
                print('  ... recalculating full adata graph')
                adata = self.graph_pp(adata)
            self.adata = adata
            
        if self.adata_out is not None:
            print('\n... saving adata:')
            adata.write(adata_out)
            print('      written to {}'.format(adata_out))
        
        print('  ... generating graphs per split after {:.1f}-s'.format(time.time() - tic))
        # no need to save adata graphs
        adatas = {}
        for split in adata.obs['split'].unique():
            tic_graph_pp_split = time.time()
            print('  ...   graph_pp() split for {}'.format(split))
            adatas[split] = self.graph_pp(adata[adata.obs['split']==split, :])  
            print('  ...     finished in {:.2f}-s'.format(time.time() - tic_graph_pp_split))
            
        if self.return_adatas or self.pkl_out is not None:
            self.adatas = adatas
            
        # get pytorch geometric objects (pgos) per split
        if self.return_pgdata or self.pkl_out is not None:
            print('  ... creating pytorch_geometric data objects after {:.1f}-s'.format(time.time() - tic))
            pg_data = {}
            for split in adata.obs['split'].unique():
                pg_data[split] = self.get_pgdata(adatas[split])
            self.pg_data = pg_data
            
        print('  loaded single-cell data in {:.1f}-min'.format((time.time() - tic)/60))
        
        if self.pkl_out is not None:
            self.pkl_that()
            
    def add_splits(self, adata, p_val=0.15, p_test=0.15):
        adata.obs['split'] = 'train'
        adata.obs.loc[adata.obs.loc[adata.obs['split']=='train', :].sample(int(p_val * adata.shape[0])).index, 'split'] = 'val'
        adata.obs.loc[adata.obs.loc[adata.obs['split']=='train', :].sample(int(p_test * adata.shape[0])).index, 'split'] = 'test'
        return adata
            
    def label_encoding(self, adata, label_colname):
        '''Give full adata so as to not miss class labels, sort (e.g., alphabetize) encoding
        
        '''
        if label_colname not in adata.obs.columns.to_list():
            warnings.warn('NEED A VALID LABEL! Dummy label will be output. skipped', UserWarning)
            return adata
        if 'y_' not in label_colname:
            # assume invalid
            print('  .. encoding {} to y_{}'.format(label_colname, label_colname))
            label_colname_new = 'y_{}'.format(label_colname)
            label_encoder = dict(zip(np.sort(adata.obs[label_colname].unique()), 
                                     np.arange(len(adata.obs[label_colname].unique()))))
            print('\n----y_{} encoding:----\n'.format(label_colname), label_encoder)
            print('----y_{} encoding:----\n'.format(label_colname))
            adata.obs[label_colname_new] = adata.obs[label_colname].map(label_encoder) 
            if self.adata_out is None:
                warnings.warn('New label encoding and label_column added to adata.obs. For reproducibility, save adata by specifying adata_out=filename_out.', UserWarning)
            self.label = label_colname_new
            return adata
        else:
            # assume all good
            self.label = label_colname
            return adata
    
    def graph_pp(self, adata, bbknn=True, k=30, n_pcs=100, add_umap=False): # NOTE: change this when not making figures
        sc.tl.pca(adata, n_comps=n_pcs)
        if bbknn:
            sc.external.pp.bbknn(adata, 
                                 n_pcs=n_pcs, 
                                 neighbors_within_batch=int(k/len(adata.obs['batch'].unique()))) # use default params
        else:
            sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=k)
        if add_umap:
            sc.tl.umap(adata)
        return adata

    def add_meld(self, adata, condition_key='batch', pos_target='SARS2'): 
        adata.obs['res'] = [1 if i==pos_target else -1 for i in adata.obs[condition_key]] # mean center?
        G = gt.Graph(data=adata.obsp['connectivities']+sparse.diags([1]*adata.shape[0],format='csr'),
                     precomputed='adjacency',
                     use_pygsp=True)
        G.knn_max = None
        adata.obs['meld_score'] = meld.MELD().fit_transform(G=G, RES=adata.obs['res'])
        return adata  
        
    def get_pgdata(self, adata, gene_ranger=True, matrix_ranger=False): 
        '''Gets pytorch geometric data object.
        
        TODO (ngr): add edge features here

        Arguments:
          adata (sc.AnnData): a dict of sc.AnnData objects specifying split

        Returns:
          adata (dict): where (key, value) is (split, Pytorch Geometric Data object)
        '''
        # feature matrix
        if gene_ranger:
            minimum = adata.X.min(axis=0)
            maximum = adata.X.max(axis=0)
            num = adata.X - minimum.todense()
            denom =  (maximum - minimum).todense()
            xhat = np.divide(num, denom, out=np.zeros_like(num), where=denom!=0) 
        elif matrix_ranger:
            # matrix in [0,1]; F.softmax(adata.X, dim=1)?
            xhat = (adata.X - adata.X.min()) / (adata.X.max() - adata.X.min()) # sc.AnnData.todense()?
        else:
            if not isinstance(adata.X, np.ndarray):
                xhat = np.asarray(adata.X.todense()) # memory intensive
            else:
                xhat = np.asarray(adata.X) # for BN later 

        # adj
        adj = adata.obsp['connectivities'] + sparse.diags([1]*adata.shape[0], format='csr')
        return Data(x=torch.from_numpy(xhat).float(), edge_index=torch.LongTensor(adj.nonzero()), 
                    y=torch.LongTensor(adata.obs[self.label]))
    
    def pkl_that(self):
        data_asdict = {
            'adata':self.adata,
            'adatas':self.adatas,
            'pg_data':self.pg_data
        }
        with open(self.pkl_out, 'wb') as f:
            pickle.dump(data_asdict, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
            
    def load_datapkl(self, filename):
        with open(filename, 'rb') as f:
            X = pickle.load(f)
            f.close()
        return X
            
    
    
## minibatching scheme
def minibatcher(d, batch_size=None):
    if batch_size is None:
        batch_size = int((np.sqrt(d.y.shape[0]))/32)
    cd = ClusterData(d,num_parts=int(np.sqrt(d.y.shape[0])))
    return ClusterLoader(cd,batch_size=batch_size, shuffle=True)
    
            
    
            
if __name__ == '__main__':
    pkl_out = '/home/ngr4/project/scnd/data/processed/mouse_220808_model_data.pkl'
    adata_out = '/home/ngr4/project/scnd/data/processed/mouse_220808.h5ad'
    adata_in = '/home/ngr4/project/scnd/data/mouse_220805.h5ad'
    
    data = pp_singlecell_data(
        adata_filename=adata_in, label_colname=['genotime_crct', 'genotype_crct'], 
        pkl_out='/home/ngr4/project/scnd/data/processed/mouse_220808_model_data.pkl', 
        adata_out='/home/ngr4/project/scnd/data/processed/mouse_220808.h5ad', 
        return_joint_adata=True, return_adatas=True, return_pgdata=True)

