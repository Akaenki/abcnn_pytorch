# ABCNN_pytorch
[Attention-Based Convolutional Neural Network for Modeling Sentence Pairs](https://arxiv.org/abs/1512.05193)

## Usage

 - Need Data and Dataloader for that data
 - Clone the repository.
 - Run `pip3 install -r requirements.txt` to install project dependencies.
 - to use, run  `python3 main.py`.

## File descriptions
```bash
├── README.md
├── sample_data/ # empty directory because of license
├── abcnn.py # model
├── dataset.py # data load
├── main.py
├── options.toml # options
├── requirements.txt
└── train.py # training function
```

##### Options
```bash
[model]
embeddeddimension = 200 # embedding vector size
strlenmax = 15  #sentence length
filterwidth = 1
filterchannel = 130
layersize = 2
inception = true # variety receptive field
distance = 'cosine' # cosine or manhattan
```

## Dependencies
 - JPype1==0.6.2
 - JPype1-py3==0.5.5.2
 - konlpy==0.4.4
 - mecab-python===0.996-ko-0.9.0
 - numpy==1.14.2
 - toml==0.9.4
 - torch==0.4.0

jype1, konlpy, mecab are for korean dataset
you don't have to use dataset.py and these libraries.

## Note
 - I used pretrained word2vec
 - I used this model to predict question similarity
 - Batch Norm makes learning faster
 - Maximum layer size is 2 in paper. Plain model cannot be learned if layer size is over 10, but model with inception module can be learned and better than shallower