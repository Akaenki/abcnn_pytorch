import numpy as np
import torch
from torch import nn, optim
from dataset import KinQueryDataset, preprocess
from torch.utils.data import DataLoader

from abcnn import Abcnn, weights_init

def train(options):
    device = torch.device("cuda" if options['general']['usecudnn'] else "cpu")

    #batch_size, emb_dim, sentence_length, filter_w, filter_c=100, layer_size=2
    model = Abcnn(options['model']['embeddeddimension'],
                options['model']['strlenmax'],
                options['model']['filterwidth'],
                options['model']['filterchannel'],
                options['model']['layersize'],
                match=options['model']['distance'],
                inception=options['model']['inception'])
    model.apply(weights_init)

    if(options['general']['loadpretrainedmodel']):
        model.load_state_dict(torch.load(options['general']['pretrainedmodelpath']))

    model = model.to(device)

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
            x1 = torch.from_numpy(x1).float()
            x2 = torch.from_numpy(x2).float()
            
            x1 = x1.to(device)
            x2 = x2.to(device)
            
            label_vars = torch.from_numpy(labels)
            label_vars = label_vars.to(device)

            predictions = model(x1, x2)
            
            loss = criterion(predictions.float(), label_vars)
            loss = loss.to(device)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        print('epoch:', epoch, ' train_loss:', float(avg_loss/total_batch))
        if options['general']['savemodel']:
            torch.save(model.state_dict(), options['general']['pretrainedmodelpath'])