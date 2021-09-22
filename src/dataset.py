import torch as t
import torch
from torch import nn
import torch.nn.functional as F
import scipy
from copy import deepcopy
import numpy as np
import pandas as pd
import sys
import math
import pickle as pkl
import math
import torch as t 
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
from sklearn.metrics import precision_score,f1_score
from torch_geometric.utils import to_undirected,remove_self_loops
from torch.nn.init import xavier_normal_,kaiming_normal_
from torch.nn.init import uniform_,kaiming_uniform_,constant
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.data import Batch,Data
from collections import Counter 
from torch.utils import data as tdata
from sklearn.model_selection import StratifiedKFold



def collate_func(batch):
    data0 = batch[0]
    if isinstance(data0,Data):
        tmp_x = [xx['x'] for xx in batch]
        tmp_y = [xx['y'] for xx in batch]
        # tmp_data = Data()
        # tmp_data['x']= t.stack(tmp_x,dim=1)
        # tmp_data['y']= t.cat(tmp_y) # 
        # tmp_data['edge_index']=data0.edge_index 
        # return Batch.from_data_list([tmp_data])
    elif isinstance(data0,(list,tuple)):
        tmp_x = [xx[0] for xx in batch]
        tmp_y = [xx[1] for xx in batch]

    tmp_data = Data()
    tmp_data['x']= t.stack(tmp_x,dim=1)
    tmp_data['y']= t.cat(tmp_y) # 
    tmp_data['edge_index']=data0.edge_index 
    tmp_data['batch'] = t.zeros_like(tmp_data['y'])
    tmp_data['num_graphs'] = 1 
    return tmp_data
    # return Batch.from_data_list([tmp_data])


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 **kwargs):
        if 'collate_fn' not in kwargs.keys():
            raise
            
        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle, **kwargs)


class ExprDataset(tdata.Dataset):#需要继承data.Dataset
    def __init__(self,Expr,edge,y,device='cuda',):
        super(ExprDataset, self).__init__()

        print('processing...')
        self.gene_num = Expr.shape[1]
        if isinstance(edge,list):
            print('multi graphs:',len(edge))
            self.edge_num = [x.shape[1] for x in edge]

            self.common_edge =[t.tensor(x).long().to(device) if not isinstance(x,t.Tensor) else x for x in edge]

        elif isinstance(edge,(np.ndarray,t.Tensor)):
            print('only has 1 graph.')
            self.edge_num = edge.shape[1]
            self.common_edge = edge if isinstance(edge,t.Tensor) else  t.tensor(edge).long().to(device)


        self.Expr = Expr
        self.y = y
        self.num_sam  = len(self.y)
        self.sample_mapping_list = np.arange(self.num_sam)

        if len(self.Expr.shape) ==2:
            self.num_expr_feaure  = 1
        else:
            self.num_expr_feaure = self.Expr.shape[2]

      
    def duplicate_minor_types(self,dup_odds=50,random_seed=2240):

        counter = Counter(self.y)
        max_num_types = max(counter.values() )
        impute_indexs = np.arange(self.num_sam).tolist()
        np.random.seed(2240)
        for lab in np.unique(self.y):
            # print('123,',max_num_types,np.sum(self.y==lab),dup_odds)
            if max_num_types/np.sum(self.y==lab) >dup_odds:
                impute_size = int(max_num_types/dup_odds) - np.sum(self.y==lab)
                print('duplicate #celltype %d with %d cells'%(lab,impute_size))
                # print(impute_size)
                impute_idx = np.random.choice(np.where(self.y==lab)[0],size=impute_size,replace=True).tolist()
                impute_indexs += impute_idx
        impute_indexs = np.random.permutation(impute_indexs)
        print('org/imputed #cells:',self.num_sam,len(impute_indexs))
        print('imputed amounts of each cell types',Counter(self.y[impute_indexs]))
        self.num_sam = len(impute_indexs)
        self.sample_mapping_list = impute_indexs
        

    def __getitem__(self, idx):
        if isinstance(idx, int):

            idx = self.sample_mapping_list[idx]
            data = self.get(idx)
            return data
        raise IndexError(
            'Only integers are valid '
            'indices (got {}).'.format(type(idx).__name__))
        pass

    def split(self,idx):
        return ExprDataset(self.Expr[idx,:],self.common_edge,self.y[idx],)
    
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.num_sam 

    def get(self,index):
        data = Data()
        data['x']= t.tensor(self.Expr[index,:].reshape([-1,self.num_expr_feaure])).float()
        data['y']= t.tensor(self.y[index].reshape([1,1])).long()  # 
        data['edge_index']=self.common_edge 
        
        # data.to(device)
        return data  