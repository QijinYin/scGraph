# scGraph
ScGraph is a GNN-based automatic cell identification algorithm leveraging gene interaction relationships to enhance the performance of the cell type identification.

# Requirements

- python = 3.6.7
- pytorch = 1.1.0
- pytorch-geometric = 1.3.1
- sklearn

# Installation

Download scGraph by

```shell
git clone https://github.com/QijinYin/scGraph
```

Installation has been tested in a Linux platform with Python3.6.

# Instructions

There is a demo including preprocessing and model training in ``src/demo.ipynb`` file.


## Preprocessing data for model training

```shell
python gen_data.py <options> -expr <expr_mat_file> -label <expr_label_file> -net <network_backbone_file>  -out <outputfile>
```
```
Arguments:
  expr_mat_file: scRNA-seq expression matrix with genes as rows and cells as columns (csv format)
  e.g.   EntrezID,barocode1,barocode2,barocode3,barocode3
          5685,1,0,0,0
          5692,0,0,0,0
          6193,0,0,0,1
  expr_label_file: cell types assignments (csv format)
  e.g. Barcodes ,label
        barocode1, celltype1
        barocode2, celltype1
        barocode3, celltype2
        barocode4, celltype3
  
  network_backbone_file: gene interactin network backbone (csv format)
  e.g. STRING database (in Entrez ID format)
      protein1, protein2,combined_score
      23521,6193,999
      5692,5685,999
      5591,2547,999
      6222,25873,999
  
  outputfile: preprocessed data for model training (npz format)
 
Options:
  -q <float> the top q quantile of network edges are used (default: 0.99 for STRING database)
```

## Run scGraph model

```shell
python scGraph.py -in <inputfile> -out-dir <outputfolder> -bs <batch_sie>
```

```
 Arguments:  
  inputfile: preprocessed data for model training (npz format)  
  outputfolder: the folder in which prediction results are saved 
  batch_sie : batch size for model training
```

# License

This project is licensed under the MIT License - see the LICENSE.md file for details
