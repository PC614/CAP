import os
import torch
import torchvision
from torch.autograd import Variable as V
from torch import nn
from torchvision import transforms as T



def bit_depth_red(X_before,depth):
    r=256/(2**depth)
    x_quan=torch.round(X_before*255/r)*r/255 
    return x_quan