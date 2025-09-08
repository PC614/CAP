import torch
import numpy as np
import scipy.stats as st
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import torch.nn as nn
from torch.autograd import Variable


import math
import os
import pickle
from tqdm import tqdm
import torch_dct as dct
import random
loss_fn = nn.CrossEntropyLoss()
epsilon = 16/255
alpha = 1.6/255
CUBLAS_STATUS_ALLOC_FAILED = 0
mesh_width=3
mesh_height=3
noise_scale=2
device = None
def decowa(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, M=5,num_iter=10, num_block=3,decay=1.0,mesh_width=3, mesh_height=3, rho=0.01):
    
    x = x.detach().clone()
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        grad_m = 0
        for n in range(M):
            
            adv = (x + x_adv).clone().detach()
            noise_map_hat = update_noise_map(model,adv, y)
            vwt_x = vwt(x + x_adv, noise_map_hat)
            
            out = model(vwt_x)  
            loss = loss_fn(out, y)
            loss.backward()
            grad_m += x_adv.grad.detach()
        grad = grad_m / M
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)      
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv


def vwt( x, noise_map):
    device = None
    n, c, w, h = x.size()
    X = grid_points_2d(mesh_width, mesh_height, device)
    Y = noisy_grid(mesh_width, mesh_height, noise_map, device)
    tpsb = TPS(size=(h, w), device=device)
    warped_grid_b = tpsb(X[None, ...], Y[None, ...])
    warped_grid_b = warped_grid_b.repeat(x.shape[0], 1, 1, 1)
    vwt_x = torch.grid_sampler_2d(x, warped_grid_b, 0, 0, False)
    return vwt_x
    
def update_noise_map(model,x, label):
    x.requires_grad = False
    noise_map = (torch.rand([mesh_height - 2, mesh_width - 2, 2]) - 0.5) * noise_scale
    for _ in range(1):
        noise_map.requires_grad = True
        vwt_x = vwt(x, noise_map)
        logits = model(vwt_x)
        loss = gloss_fn(logits, label)
        
        loss.backward()
        grad = noise_map.grad.detach()
       
        noise_map = noise_map.detach() - rho * grad  
    return noise_map.detach()




def K_matrix(X, Y):
    eps = 1e-9

    D2 = torch.pow(X[:, :, None, :] - Y[:, None, :, :], 2).sum(-1)
    K = D2 * torch.log(D2 + eps)
    return K

def P_matrix(X):
    n, k = X.shape[:2]
    device = X.device

    P = torch.ones(n, k, 3,device=device)
    P[:, :, 1:] = X
    return P


class TPS_coeffs(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X, Y):

        n, k = X.shape[:2]  # n = 77, k =2
        device = X.device

        Z = torch.zeros(1, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device) # [1, 80, 80]
        K = K_matrix(X, X)

        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        Q = torch.solve(Z, L)[0]
        #Q = torch.linalg.solve(L, Z)
        return Q[:, :k], Q[:, k:]

class TPS(torch.nn.Module):
    def __init__(self, size: tuple = (256, 256), device=None):
        super().__init__()
        h, w = size
        self.size = size
        self.device = device
        self.tps = TPS_coeffs()
        grid = torch.ones(1, h, w, 2, device=device)
        grid[:, :, :, 0] = torch.linspace(-1, 1, w)
        grid[:, :, :, 1] = torch.linspace(-1, 1, h)[..., None]
        self.grid = grid.view(-1, h * w, 2)

    def forward(self, X, Y):
        """Override abstract function."""
        h, w = self.size
        W, A = self.tps(X, Y)  
        U = K_matrix(self.grid, X) 
        P = P_matrix(self.grid)
        grid = P @ A + U @ W
        return grid.view(-1, h, w, 2) 

def grid_points_2d(width, height, device):
    xx, yy = torch.meshgrid(
        [torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device)])
    return torch.stack([yy, xx], dim=-1).contiguous().view(-1, 2)

def noisy_grid(width, height, noise_map, device):
    """
    Make uniform grid points, and add noise except for edge points.
    """
    grid = grid_points_2d(width, height, device)
    mod = torch.zeros([height, width, 2], device=device)
    mod[1:height - 1, 1:width - 1, :] = noise_map
    return grid + mod.reshape(-1, 2)
