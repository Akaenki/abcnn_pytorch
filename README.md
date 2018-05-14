# ABCNN_pytorch
[Attention-Based Convolutional Neural Network for Modeling Sentence Pairs](https://arxiv.org/abs/1512.05193)

## Usage
 - Install [Python 3].
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
[general]

```

## Dependencies
 - JPype1==0.6.2
 - JPype1-py3==0.5.5.2
 - konlpy==0.4.4
 - mecab-python===0.996-ko-0.9.0
 - numpy==1.14.2
 - toml==0.9.4
 - torch==0.4.0
 - torchvision==0.2.1

jype1, konlpy, mecab are for korean dataset
you don't have to use dataset.py. it's for korean dataset

## Note