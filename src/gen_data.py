import numpy as np 
import pandas as pd 
import argparse
from scipy.io import mmread


# +

def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-expr','--expr',type=str,)
    parser.add_argument('-label','--label',type=str,)
    parser.add_argument('-net','--net',type=str,)
    parser.add_argument('-out','--outfile',type=str,)
    parser.add_argument('-q','--quantile',type=float,default='0.99')
    return parser



def add_remaining_self_loop_for_edge_df(edge_df,
                             edge_weight_column = 'score',
                             fill_value = 1.,
                             num_nodes= None):

    '''
    edge_df : #num_edges x 2
    
    '''
    assert 'node1' in edge_df.columns
    assert 'node2' in edge_df.columns
    edge_index = edge_df[['node1','node2']].T.values
    row, col = edge_index[0], edge_index[1]
    N = num_nodes if num_nodes is not None else np.max(edge_index)+1
    
    mask = row == col
    added_index = list(set( np.arange(0, N, dtype=int ))-set(row[mask]))
    
    new_df = pd.DataFrame()
    new_df['node1'] = added_index
    new_df['node2'] = added_index
    
    if edge_weight_column in edge_df.columns:
        new_df[edge_weight_column] = fill_value
        
    edge_df = edge_df.append(new_df, ignore_index=True)
    return edge_df


def coding_edge_with_ref_gene_idx(converted_graph_gene,converted_expr_gene):
    '''
    converted_graph_gene: #2 Dim 
    converted_expr_gene: #1 Dim 
    convert graph_gene to index which is the order of converted_expr_gene and drop nan
    '''
    graph_edge_df = pd.DataFrame(converted_graph_gene,columns=['node1','node2'])
    gene_to_index_dict = {g:idx for idx,g in enumerate(converted_expr_gene)}
    
    mapfunc = lambda x:  gene_to_index_dict.get(x,np.nan)
    graph_edge_df  = graph_edge_df.applymap(mapfunc)
    graph_edge_df = graph_edge_df.dropna(axis='index')
    
#     print(graph_edge_df.shape)
    graph_edge_df = add_remaining_self_loop_for_edge_df(graph_edge_df,edge_weight_column='score',fill_value=1,num_nodes=len(converted_expr_gene))
#     print(graph_edge_df.shape)
    return graph_edge_df.values



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print('args:',args)

    expr_file = args.expr 
    label_file = args.label 
    net_file = args.net 
    thres = args.quantile 
    save_file = args.outfile
    assert 0<=thres<=1,"quantile should be a float value in [0,1]."


    data_df = pd.read_csv(expr_file,header=0,index_col=0)
    label_df = pd.read_csv(label_file,header=0,index_col=0)

    graph_df = pd.read_csv(net_file,header=None,index_col=None,)
    graph_df.columns = ['node1', 'node2', 'score']
    graph_df = graph_df.loc[graph_df.score.ge(graph_df.score.quantile(0.99)).values,['node1','node2']] # quantile 0.99


    # normalize + log1p transform for read counts 
    data_df = data_df.apply(lambda x: 1e6* x/x.sum()+1e-5,axis=0)
    data_df = data_df.applymap(np.log1p)


    str_labels = np.unique(label_df.values).tolist()
    label = [str_labels.index(x) for x in label_df.values ]
    gene = data_df.index.values
    barcode = data_df.columns.values
    edge_index  = coding_edge_with_ref_gene_idx(graph_df.values,gene)

    print('shape of expression matrix [#genes,#cells]:',data_df.shape)
    print('shape of cell labels:',len(label))
    print('number of cell types:',len(str_labels))
    print('shape of backbone network:',edge_index.shape)


    data_dict = {}
    data_dict['barcode'] = barcode
    data_dict['gene'] = gene
    data_dict['logExpr'] = data_df.values 
    data_dict['str_labels'] = str_labels
    data_dict['label'] = label
    data_dict['edge_index'] = edge_index 

    np.savez(save_file,**data_dict)

    print('Finished.')