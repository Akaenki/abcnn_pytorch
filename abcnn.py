import numpy as np
import torch
from torch.nn import functional as F

class Abcnn(nn.Module):

    def __init__(self, emb_dim, sentence_length, filter_width, filter_channel=100, layer_size=2, match='cosine', inception=True):
        super(Abcnn, self).__init__()
        self.layer_size = layer_size
            
        if match == 'cosine':
            self.match = cosine_similarity
        else:
            self.match = manhattan_distance

        self.ap_input = ApLayer(sentence_length, emb_dim)
        self.attention_bn = nn.ModuleList()
        self.abcnn_layers = nn.ModuleList()
        self.conv = nn.ModuleList()
        self.ap = nn.ModuleList()
        self.wp = WpLayer(sentence_length, filter_width)
        self.fc = nn.Linear(layer_size+1, 1)

        for i in range(layer_size):
            self.attention_bn.append(nn.BatchNorm2d(2))
            self.abcnn_layers.append(Abcnn1(sentence_length, emb_dim if i == 0 else filter_c))
            self.conv.append(ConvLayer(sentence_length, filter_width, emb_dim if i == 0 else filter_channel, filter_channel, inception))
            self.ap.append(ApLayer(sentence_length + filter_width - 1, filter_channel))
        
    def forward(self, x1, x2):
        sim = []
        
        sim.append(self.match(self.ap_input(x1), self.ap_input(x2)))
        for i in range(self.layer_size):
            if len(list(x1.size())) == 3:
                x1 = x1.unsqueeze(1)
                x2 = x2.unsqueeze(1)
            a_matrix = attention_matrix(x1, x2)
            x1 = self.abcnn_layers[i](x1, a_matrix.permute(0, 2, 1))
            x2 = self.abcnn_layers[i](x2, a_matrix)
            x1 = self.attention_bn[i](x1)
            x2 = self.attention_bn[i](x2)
            x1 = self.conv[i](x1)
            x2 = self.conv[i](x2)
            sim.append(self.match(self.ap[i](x1), self.ap[i](x2)))
            a_conv = attention_matrix(x1, x2)
            x1_a_conv = a_conv.sum(dim=1)
            x2_a_conv = a_conv.sum(dim=2)
            x1 = self.wp(x1, x1_a_conv)
            x2 = self.wp(x2, x2_a_conv)
        sim_fc = torch.cat(sim, dim=1)
        output = self.fc(sim_fc)
        return output

class InceptionModule(nn.Module):
    
    def __init__(self, strmaxlen, filter_width, filter_height, filter_channel):
        super(InceptionModule,self).__init__()
        self.conv_1 = convolution(filter_width, filter_height, int(filter_channel/3) + filter_channel - 3*int(filter_channel/3), filter_width-1)
        self.conv_2 = convolution(filter_width+4, filter_height, int(filter_channel/3), filter_width+1)
        self.conv_3 = convolution(strmaxlen, filter_height, int(filter_channel/3), int((strmaxlen+filter_width-2)/2))

    def forward(self, x):
        out_1 = self.conv_1(x)
        out_2 = self.conv_2(x)
        out_3 = self.conv_3(x)
        output = torch.cat([out_1, out_2, out_3], dim=1)
        return output

class Abcnn1(nn.Module):
    '''
    attention matrix = (bs, w, w)
    x = (bs, 1, w, h)
    x_a = (bs, 1, w, h)
    output = (bs, 2, w, h)
    '''
    def __init__(self, in_dim, out_dim):
        super(Abcnn1, self).__init__()
        self.attention_feature_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x, attention_matrix):
        x_a = self.attention_feature_layer(attention_matrix)
        x_a = x_a.unsqueeze(1)
        return torch.cat([x, x_a], 1)

class ConvLayer(nn.Module):

    def __init__(self, strmaxlen, filter_width, filter_height, filter_channel, inception=True):
        super(ConvLayer, self).__init__()
        if inception:
            self.model = InceptionModule(strmaxlen, filter_width, filter_height, filter_channel)
        else:
            self.model = convolution(filter_width, filter_height, filter_channel, filter_width-1)

    def forward(self, x):
        output = self.model(x)
        output = output.permute(0, 3, 2, 1)
        return output

def cosine_similarity(x1, x2):
    '''
    x = (bs, h)
    after cos = (bs,)
    output = (bs, 1)
    '''
    return F.cosine_similarity(x1, x2).unsqueeze(1)

def manhattan_distance(x1, x2):
    '''
    x = (bs, h)
    output = (bs, 1)
    '''
    return torch.div(torch.norm((x1 - x2), 1, 1, keepdim=True), x1.size()[1])

def convolution(filter_width, filter_height, filter_channel, padding):
    
    model = nn.Sequential(
        nn.Conv2d(2, filter_channel, (filter_width, filter_height), stride=1, padding=(padding, 0)),
        nn.BatchNorm2d(filter_channel),
        nn.Tanh()
    )
    return model
    

def attention_matrix(x1, x2):
    '''
    make attention matrix using match score
    x = (bs, 1(channel), w, h)
    permuted x = (bs, w, 1, h)
    output = (bs, w(for x2), w(for x1))
    '''
    
    eps = Variable(torch.Tensor([1e-6]))
    one = Variable(torch.Tensor([1]))
    try:
        eps = eps.cuda()
        one = one.cuda()
    except:
        pass
    euclidean = (((x1 - x2.permute(0, 2, 1, 3)) ** 2).sum(dim=3) + eps).sqrt()
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
        nn.init.xavier_uniform(m.weight)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform(m.weight)
        nn.init.constant(m.bias, 0.1)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)