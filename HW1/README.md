# Homework 1 by Natural Language Processors

This is our code for HW1. This repo includes the script for training out models from scratch as well as the files 
needed to restore the models as we have trained them.

## CNN models
We found that batch sizes of 50-100 worked much better for training than batch sizes of 10, therefore we broke up 
model training and prediction writing into two scripts.

Also please run
```
mkdir ./data
mkdir ./src
```
and place `cnn.py` and `cnn_eval.py` in `src` dir. Predictions and models are saved to `data`.
## Running the code
To initialize the agent with our trained model and run greedily, with no further learning:
```
python cnn.py [normal, extralayers, dilated, multi] [True, False] [True, False]
```
The script accepts up to three positional arguments. If "normal", "extralayers", or "dilated" are used in the first position, no other arguments 
should be given. If "multi" is specified in the first position, two additional arguments must be provided.

### normal
This model is our implementation of Yoon's published model with dynamic embeddings.
### extralayers
This is the same as the above model, except, an average pooling and additional convolution have been added after
the multi-width conv layer and before the adaptive maxpool layer
### dilated
Similar to extralayers, except extra convolution is dilated
### multi
The Multi-channel CNN model in Yoon's paper. We have added the option for GloVe or word2vec to be used in the static or 
dynamic channels. The first True/False flag sets the dynamic embedding, the second True/False flag sets the static embedding 
True indicates GloVe should be used, False indicates word2vec should be used.

## Evaluating CNN models
To evaluate a trained model, use `cnn_eval.py` in place of `cnn.py` with the same aruments
```
python cnn_eval.py [normal, extralayers, dilated, multi] [True, False] [True, False]
```
After evaluating models, and getting lackluster results, we decided to use a simple ensemble to improve accuracy. We collected predictions from the one `normal` and 4 `multi` models and used a majority vote to decide category assignment.
This can be reproduced with the following widget, and `cnn_ensemble1_predictions.csv` will be generated in `data`:
```
python voting_ensemble.py ../data/cnn_multi_False_False_predictions.csv \
  ../data/cnn_multi_False_True_predictions.csv \
  ../data/cnn_multi_True_False_predictions.csv \
  ../data/cnn_multi_True_True_predictions.csv \
  ../data/cnn_normal_predictions.csv
 ```
