# bipkt



## Quick Start

conda create --name=bipkt python=3.7.5 

source activate bipkt

pip install -U pykt-toolkit

## Prepare a Dataset
1、Obtain a Dataset
Let’s start by downloading the dataset from here. Please make sure you have creat the data/{dataset_name} folder


2、Data Preprocessing

python data_preprocess.py [parameter]

Args:
    
--dataset_name: dataset name, default=“assist2015”
    
--min_seq_len: minimum sequence length, default=3
    
--maxlen: maximum sequence length, default=200
    
--kfold: divided folds, default=5

Example:

cd examples 

python data_preprocess.py --dataset_name=lg

## Training a Model
After the data preprocessing, you can use the python wandb_modelname_train.py [parameter] to train a model:

CUDA_VISIBLE_DEVICES=2 nohup python wandb_bipkt_train.py --dataset_name=lg --use_wandb=0 --add_uuid=0 

## Evaluating Your Model

ow, let’s use wandb_predict.py to evaluate the model performance on the testing set.

python wandb_predict.py

Args:
    
--bz: batch_size, default is 256
    
--save_dir: the dictory of the trained model, default is "saved_model"
    
--fusion_type: the fusion mode,default is "late_fusion"
    
--use_wandb: use wandb or not, default is 1
