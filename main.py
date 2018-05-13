from __future__ import print_function
import torch
import toml
from train import train

print('Loading options...')
with open('options.toml', 'r') as optionsFile:
    options = toml.loads(optionsFile.read())

if(options['general']['usecudnnbenchmark'] and options['general']['usecudnn']):
    print('Running cudnn benchmark...')
    torch.backends.cudnn.benchmark = True

train(options)