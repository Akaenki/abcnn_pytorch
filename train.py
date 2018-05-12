import numpy as np
import torch
from torch import nn, optim
from dataset import KinQueryDataset, preprocess
from torch.utils.data import DataLoader
from torch.autograd import Variable

from abcnn import Abcnn, weights_init

def train(options):

    #batch_size, emb_dim, sentence_length, filter_w, filter_c=100, layer_size=2
    model = Abcnn(options['model']['embeddeddimension'],
                options['model']['strlenmax'],
                options['model']['filterwidth'],
                options['model']['filterchannel'],
                options['model']['layersize'],
                match=options['model']['matchscore'],
                inception=options['model']['inception'])
    model.apply(weights_init)

    if(options['general']['loadpretrainedmodel']):
        model.load_state_dict(torch.load(options['general']['pretrainedmodelpath']))

    if options['general']['usecudnn']:
        model = model.cuda(options['general']['gpuid'])

    optimizer = optim.Adam(model.parameters(),
                    lr=options['training']['learningrate'],
                    weight_decay=options['training']['weightdecay'])
    criterion = nn.BCEWithLogitsLoss()

    dataset = KinQueryDataset(options['training']['dataset'], options['model']['strlenmax'], options['model']['embeddeddimension'])
    train_loader = DataLoader(dataset=dataset,
                        batch_size=options['input']['batchsize'],
                        shuffle=options['input']['shuffle'],
                        num_workers=options['input']['numworkers'])
    total_batch = len(train_loader)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    for epoch in rrange(options['training']['startepoch'], options['training']['epochs']):
        avg_loss = 0.0
        scheduler.step()
        
        for data, labels in train_loader:
            data = np.array(data)
            x1 = data[:, 0, :, :]
            x2 = data[:, 1, :, :]
            x1 = Variable(torch.from_numpy(x1).float())
            x2 = Variable(torch.from_numpy(x2).float())
            
            if options['general']['usecudnn']:
                x1 = x1.cuda(options['general']['gpuid'])
                x2 = x2.cuda(options['general']['gpuid'])
                
            predictions = model(x1, x2)
            label_vars = Variable(torch.from_numpy(labels))

            if options['general']['usecudnn']:
                label_vars = label_vars.cuda(options['general']['gpuid'])
            
            loss = criterion(predictions.float(), label_vars)
            
            if options['general']['usecudnn']:
                loss = loss.cuda(options['general']['gpuid'])
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.data[0]

        print('epoch:', epoch, ' train_loss:', float(avg_loss/total_batch))
        if options['general']['savemodel']:
            torch.save(model.state_dict(), options['general']['pretrainedmodelpath'])