from PIL import Image
import os
import torch
import torchvision
from torch.autograd import Variable as V
from torch import nn
from torchvision import transforms as T

def JPEG_compression(X_before,quality):
        trn = T.Compose([T.ToTensor()])
        X_after=torch.zeros_like(X_before)
        for j in range(X_after.size(0)):
            x_np=T.ToPILImage()(X_before[j].detach().cpu())
            x_np.save('./'+'j.jpg',quality=quality)
            X_after[j]=trn(Image.open('./'+'j.jpg'))
        return X_after