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



import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
import math
def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=False, bias=True,activate=False,alphas=[0,1],shared_weight=False,aggr = 'mean',
                 **kwargs):
        super(SAGEConv, self).__init__(aggr=aggr, **kwargs)
        print('SAGEConv activate,aggr',activate,aggr)
        self.shared_weight = shared_weight
        self.activate = activate
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        if self.shared_weight:
            print('shared weights for SAGEConv2')
            self.self_weight = self.weight 
        else:
            self.self_weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        self.alphas = alphas #[self_alpha, pro_alpha]
        print('alphas',self.alphas)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)
        uniform(self.in_channels,self.self_weight)

    def forward(self, x, edge_index, edge_weight=None, size=None):

        out  =  torch.matmul(x,self.self_weight )
        out2 = self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight)
        return self.alphas[0]*out+ self.alphas[1]* out2

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1,1) * x_j

    def update(self, aggr_out):

        if self.activate:
            aggr_out = F.relu(aggr_out)
            
        if torch.is_tensor(aggr_out):
            aggr_out = torch.matmul(aggr_out,self.weight )
        else:
            aggr_out = (None if aggr_out[0] is None else torch.matmul(aggr_out[0], self.weight),
                 None if aggr_out[1] is None else torch.matmul(aggr_out[1], self.weight))
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



from torch.nn.init import normal,uniform_
def init_weights(m):
    if type(m) == nn.Linear:
        # nn.init.kaiming_normal_(m.weight, mode='fan_out')
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def help_bn(bn1,x):
    #         dim1, dim2,dim3 = x.shape 
    x = x.permute(1,0,2)  # #samples x #nodes x #features
    x = bn1(x)
    x = x.permute(1,0,2) # #nodes x #samples x #features
    return x

class scGraph(nn.Module):
    def __init__(self,in_channel=1,mid_channel=8,out_channel=2,num_nodes=2207,edge_num=151215,
                **args):
        super(scGraph,self).__init__()
        # self.max_pool_dim = 100 

        self.mid_channel = mid_channel
        self.dropout_ratio = args.get('dropout_ratio',0.3)
        print('model dropout raito:',self.dropout_ratio)
        n_out_nodes = num_nodes
        self.global_conv1_dim = 4*3
        self.global_conv2_dim = args.get('global_conv2_dim',4)
        self.conv1 = SAGEConv(in_channel, mid_channel, )
        self.bn1 = torch.nn.LayerNorm((num_nodes,mid_channel))
        self.act1 = nn.ReLU()

        self.global_conv1 = t.nn.Conv2d(mid_channel*1,self.global_conv1_dim,[1,1])
        self.global_bn1 = torch.nn.BatchNorm2d(self.global_conv1_dim)
        self.global_act1 = nn.ReLU()
        self.global_conv2 = t.nn.Conv2d(self.global_conv1_dim,self.global_conv2_dim,[1,1])
        self.global_bn2 = torch.nn.BatchNorm2d(self.global_conv2_dim)
        self.global_act2 = nn.ReLU()

            
        last_feature_node = 64
        channel_list = [self.global_conv2_dim*n_out_nodes,256,64]
        if args.get('channel_list',False):
            channel_list = [self.global_conv2_dim*n_out_nodes,128]
            last_feature_node = 128

        self.nn = []
        for idx,num in enumerate(channel_list[:-1]):
            self.nn.append(nn.Linear(channel_list[idx],channel_list[idx+1]))
            self.nn.append(nn.BatchNorm1d(channel_list[idx+1]))
            if self.dropout_ratio >0:
                self.nn.append(nn.Dropout(0.3))
            self.nn.append(nn.ReLU())
        self.global_fc_nn =nn.Sequential(*self.nn)        
        # self.fc1 = nn.Linear(self.global_conv1_dim*self.max_pool_dim, 128)
        self.fc2 = nn.Linear(last_feature_node,out_channel)
        
        self.edge_num = edge_num
        self.weight_edge_flag = True  
        print('trainalbe edges :',self.weight_edge_flag)
        if self.weight_edge_flag:
            self.edge_weight = nn.Parameter(t.ones(edge_num).float()*0.01)
            print(self.edge_weight.shape)
            # _=normal(self.edge_weight,mean=0,std=0.01)
            # _ = uniform_(self.edge_weight,a=-2,b=2)
            print(self.edge_weight[:10])
        else:
            self.edge_weight = None
    
        self.reset_parameters()
    
    def reset_parameters(self,):
        # uniform(self.mid_channel, self.global_conv1.weight
        # 
        self.conv1.apply(init_weights)
        nn.init.kaiming_normal_(self.global_conv1.weight, mode='fan_out')
        uniform(self.mid_channel, self.global_conv1.bias)


        nn.init.kaiming_normal_(self.global_conv2.weight, mode='fan_out')
        uniform(self.global_conv1_dim, self.global_conv2.bias)

        self.global_fc_nn.apply(init_weights)
        self.fc2.apply(init_weights)
        # uniform(self.in_channels, self.bias)
        # uniform(self.in_channels,self.self_weight)
        pass

    def get_gcn_weight_penalty(self,mode='L2'):

        if mode == 'L1':
            func = lambda x:  t.sum(t.abs(x))
        elif mode =='L2':
            func  = lambda x: t.sqrt(t.sum(x**2))

        loss = 0 

        tmp = getattr(self.conv1,'weight',None)
        if tmp is not None: 
            loss += func(tmp)

        tmp = getattr(self.conv1,'self_weight',None)
        if tmp is not None: 
            loss += 1* func(tmp)

        tmp = getattr(self.global_conv1,'weight',None)
        if tmp is not None: 
            loss += func(tmp)
        tmp = getattr(self.global_conv2,'weight',None)
        if tmp is not None: 
            loss += func(tmp)

        return loss 


    def forward(self,data,get_latent_varaible=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
  
        if self.weight_edge_flag:
            one_graph_edge_weight=torch.sigmoid(self.edge_weight)#*self.edge_num
            edge_weight = one_graph_edge_weight
        else:
            edge_weight = None 

        x = self.act1(self.conv1(x, edge_index,edge_weight=edge_weight))
        x = help_bn(self.bn1,x)
        if self.dropout_ratio >0: x = F.dropout(x, p=0.1, training=self.training)
        

        x = x.permute(1,2,0)  # #samples x #features x #nodes 
        x = x.unsqueeze(dim=-1) # #samples x #features x #nodes x 1
        x = self.global_conv1(x)  # #samples x #features x #nodes x 1
        x = self.global_act1(x)
        x = self.global_bn1(x)
        if self.dropout_ratio >0: x = F.dropout(x, p=0.3, training=self.training)
        x = self.global_conv2(x)
        x = self.global_act1(x)
        x = self.global_bn2(x)
        if self.dropout_ratio >0: x = F.dropout(x, p=0.3, training=self.training)
        x = x.squeeze(dim=-1)  # #samples  x #features  x #nodes 
        num_samples = x.shape[0]

        x = x .view(num_samples,-1)
        x = self.global_fc_nn(x)
        if get_latent_varaible:
            return x
        else: 
            x = self.fc2(x)
            
            return F.softmax(x, dim=-1) 



def edge_transform_func(org_edge):
    edge = org_edge
    edge = t.tensor(edge.T)
    edge = remove_self_loops(edge)[0]
    edge = edge.numpy()
    return edge