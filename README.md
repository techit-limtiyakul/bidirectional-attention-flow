# PyTorch implementation for Bi-directional Attention Flow model 
Question answering system based on the paper https://arxiv.org/abs/1611.01603. This is a slightly modified version where its answer selection module use bilinear function, giving slight improvement in accuracy over the original model. Part of the code are from https://github.com/jojonki/BiDAF. 

In Order to run,
## 1. Download  SQuAD dataset and GloVe embeddings
(~850 MB, this will download files to `$HOME/data`):
```
chmod +x download.sh; ./download.sh
```

## 2. Preprocess SQuAD data
```
python -m squad.prepro
```

Then place the processed data and unzipped GloVe embeddings into the data directory (by default it is ./data/squad)

## 3. Training
To train, run the following command.
```
python main.py
```
To test, 
```
python main.py --test 1 --resume <PATH_TO_SAVED_PARAMS>
```
