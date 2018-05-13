import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class Abcnn3(nn.Module):

    def __init__(self, emb_dim, sentence_length, filter_width, filter_channel=100, layer_size=2, match='cosine', inception=True):
        super(Abcnn3, self).__init__()
        self.layer_size = layer_size
            
        if match == 'cosine':
            self.distance = cosine_similarity
        else:
            self.distance = manhattan_distance

        self.abcnn1 = nn.ModuleList()
        self.abcnn2 = Abcnn2Portion(sentence_length, filter_width)
        self.conv = nn.ModuleList()
        self.ap = nn.ModuleList()
        self.fc = nn.Linear(layer_size+1, 1)

        self.ap.append(ApLayer(sentence_length, emb_dim))

        for i in range(layer_size):
            self.abcnn1.append(Abcnn1Portion(sentence_length, emb_dim if i == 0 else filter_channel))
            self.conv.append(ConvLayer(sentence_length, filter_width, emb_dim if i == 0 else filter_channel, filter_channel, inception))
            self.ap.append(ApLayer(sentence_length + filter_width - 1, filter_channel))
        
    def forward(self, x1, x2):
        sim = []
        sim.append(self.distance(self.ap[0](x1), self.ap[0](x2)))

        for i in range(self.layer_size):
            x1, x2 = self.abcnn1[i](x1, x2)
            x1 = self.conv[i](x1)
            x2 = self.conv[i](x2)
            sim.append(self.distance(self.ap[i+1](x1), self.ap[i+1](x2)))
            x1, x2 = self.abcnn2(x1, x2)
            
        sim_fc = torch.cat(sim, dim=1)
        output = self.fc(sim_fc)
        return output

class Abcnn1Portion(nn.Module):
    '''Part of Abcnn1
    '''

    def __init__(self, in_dim, out_dim):
        super(Abcnn1Portion, self).__init__()
        self.batchNorm = nn.BatchNorm2d(2)
        self.attention_feature_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x1, x2):
        '''
        1. compute attention matrix
            attention_m : size of (batch_size, w, w)
        2. generate attention feature map(weight matrix are parameters of the model to be learned)
            x_attention : size of (batch_size, 1, w, h)
        3. stack the representation feature map and attention feature map
            x : size of (batch_size, 2, w, h)
        4. batch norm(not in paper)

        Parameters
        ----------
        x1, x2 : 4-D torch Tensor
            size (batch_size, 1, w, h)

        Returns
        -------
        (x1, x2) : list of 4-D torch Tensor
            size (batch_size, 2, w, h)
        '''
        attention_m = attention_matrix(x1, x2)

        x1_attention = self.attention_feature_layer(attention_m.permute(0, 2, 1))
        x1_attention = x1_attention.unsqueeze(1)
        x1 = torch.cat([x1, x1_attention], 1)

        x2_attention = self.attention_feature_layer(attention_m)
        x2_attention = x2_attention.unsqueeze(1)
        x2 = torch.cat([x2, x2_attention], 1)

        x1 = self.batchNorm(x1)
        x2 = self.batchNorm(x2)
        
        return (x1, x2)

class Abcnn2Portion(nn.Module):
    '''Part of Abcnn2
    '''

    def __init__(self, sentence_length, filter_width):
        super(Abcnn2Portion, self).__init__()
        self.wp = WpLayer(sentence_length, filter_width)

    def forward(self, x1, x2):
        '''
        1. compute attention matrix
            attention_m : size of (batch_size, w+filter_width-1, w+filter_width-1)
        2. sum all attention values for a unit to derive a single attention weight for that unit
            x_a_conv : size of (batch_size, w+filter_width-1)
        3. average pooling(w-ap)

        Parameters
        ----------
        x1, x2 : 4-D torch Tensor
            size (batch_size, 1, w+filter_width-1, h)

        Returns
        -------
        (x1, x2) : list of 4-D torch Tensor
            size (batch_size, 1, w, h)
        '''
        attention_m = attention_matrix(x1, x2)
        x1_a_conv = attention_m.sum(dim=1)
        x2_a_conv = attention_m.sum(dim=2)
        x1 = self.wp(x1, x1_a_conv)
        x2 = self.wp(x2, x2_a_conv)

        return (x1, x2)


class InceptionModule(nn.Module):
    
    def __init__(self, sentence_length, filter_width, filter_height, filter_channel):
        super(InceptionModule,self).__init__()
        self.conv_1 = convolution(filter_width, filter_height, int(filter_channel/3) + filter_channel - 3*int(filter_channel/3), filter_width-1)
        self.conv_2 = convolution(filter_width+4, filter_height, int(filter_channel/3), filter_width+1)
        self.conv_3 = convolution(sentence_length, filter_height, int(filter_channel/3), int((sentence_length+filter_width-2)/2))

    def forward(self, x):
        out_1 = self.conv_1(x)
        out_2 = self.conv_2(x)
        out_3 = self.conv_3(x)
        output = torch.cat([out_1, out_2, out_3], dim=1)
        return output

class ConvLayer(nn.Module):

    def __init__(self, sentence_length, filter_width, filter_height, filter_channel, inception):
        super(ConvLayer, self).__init__()
        if inception:
            self.model = InceptionModule(sentence_length, filter_width, filter_height, filter_channel)
        else:
            self.model = convolution(filter_width, filter_height, filter_channel, filter_width-1)

    def forward(self, x):
        output = self.model(x)
        output = output.permute(0, 3, 2, 1)
        return output

def cosine_similarity(x1, x2):
    '''compute cosine similarity between x1 and x2

    Parameters
    ----------
    x1, x2 : 2-D torch Tensor
        size (batch_size, 1)

    Returns
    -------
    distance : 2-D torch Tensor
        similarity result of size (batch_size, 1)
    '''
    return F.cosine_similarity(x1, x2).unsqueeze(1)

def manhattan_distance(x1, x2):
    '''compute manhattan distance between x1 and x2

    Parameters
    ----------
    x1, x2 : 2-D torch Tensor
        size (batch_size, 1)

    Returns
    -------
    distance : 2-D torch Tensor
        similarity result of size (batch_size, 1)
    '''
    return torch.div(torch.norm((x1 - x2), 1, 1, keepdim=True), x1.size()[1])

def convolution(filter_width, filter_height, filter_channel, padding):
    ''' make convolution layer
    '''
    model = nn.Sequential(
        nn.Conv2d(2, filter_channel, (filter_width, filter_height), stride=1, padding=(padding, 0)),
        nn.BatchNorm2d(filter_channel),
        nn.Tanh()
    )
    return model
    
def attention_matrix(x1, x2, eps=1e-6):
    '''make attention matrix using match score
    
    1/(1 + |x · y|)
    |·| is euclidean distance

    Parameters
    ----------
    x1, x2 : 4-D torch Tensor
        size (batch_size, 1, w, h)
    
    Returns
    -------
    output : 3-D torch Tensor
        match score result of size (batch_size, w(x2), w(x1))
    '''
    
    eps = torch.tensor(eps)
    one = torch.tensor(1.)
    euclidean = (torch.pow(x1 - x2.permute(0, 2, 1, 3), 2).sum(dim=3) + eps).sqrt()
    return (euclidean + one).reciprocal()

class ApLayer(nn.Module):
    '''
    Average pooling layer to calculate similarity
    after pooling x = (bs, 1, 1, height)
    output = (bs, height)
    '''
    def __init__(self, pool_width, height):
        super(ApLayer, self).__init__()
        self.ap = nn.AvgPool2d((pool_width, 1), stride=1)
        self.height = height

    def forward(self, x):
        return self.ap(x).view([-1, self.height])

class WpLayer(nn.Module):
    '''
    x = (bs, 1, w + filter_width - 1, height)
    attention_matrix = (bs, w + filter_width - 1)
    output = (bs, 1, sentence_length, height)
    '''
    def __init__(self, sentence_length, filter_width):
        super(WpLayer, self).__init__()
        self.sentence_length = sentence_length
        self.filter_width = filter_width

    def forward(self, x, attention_matrix):
        pools = []
        attention_matrix = attention_matrix.unsqueeze(1).unsqueeze(3)
        for i in range(self.sentence_length):
            pools.append((x[:, :, i:i+self.filter_width, :] * attention_matrix[:, :, i:i+self.filter_width, :]).sum(dim=2, keepdim=True))

        return torch.cat(pools, dim=2)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Layer') == -1:
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.1)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)