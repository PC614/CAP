import torch
import numpy as np
import scipy.stats as st
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad
from torch.optim import Adam
from ge_adv import accuracy_list,avg_accuracy
import matplotlib.pyplot as plt
import math
import os
import pickle
from tqdm import tqdm
import torch_dct as dct
import random
from scipy.fftpack import dct, idct
loss_fn = nn.CrossEntropyLoss()
epsilon = 16/255
alpha = 1.6/255

def fgsm(model, x, y, loss_fn, epsilon=epsilon):
    x_adv = x.detach().clone() # initialize x_adv as original benign image x
    x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad
    out = model(x_adv)  
    loss = loss_fn(out, y)
    loss.backward() # calculate gradient
    grad = x_adv.grad.detach()
    x_adv = x_adv + epsilon * grad.sign()
    return x_adv


def ifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10):
    x_adv = x
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True 
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha* grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach() 
    return x_adv

def mifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True 
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        grad = x_adv.grad.detach() 
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def mifgsmh(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(x_adv), y)
        grad1 = torch.autograd.grad(loss, x_adv, grad_outputs=torch.ones_like(loss), create_graph=True)[0]
        grad2 = torch.autograd.grad(grad1, x_adv, grad_outputs=torch.ones_like(grad1),retain_graph=True)[0]
        grad = grad1 - 0.5*grad2@(grad1/torch.norm(grad1, p=2))
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def mifgsm20(model, x, y, loss_fn, epsilon=epsilon/3, alpha=alpha/6, beta=alpha/30,num_iter=20, num=5,decay=0.7):
    loss_history = []
    loss20_history = []
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        x_copy = x_adv
        loss = loss_fn(model(x_adv), y)
        loss_history.append(loss)
        loss.backward()
        grad = x_adv.grad.detach()
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        x_tmp = x_adv
        x_adv = x_copy
        for j in range(num):
            x_adv = x_adv.detach().clone()
            x_adv.requires_grad = True
            loss = loss_fn(model(x_adv), y)
            loss.backward()
            grad = x_adv.grad.detach()
            x_adv = x_adv +  beta * grad.sign()
            delta = torch.clamp(x_adv - x_copy, min=-alpha, max=alpha)
            x_adv = torch.clamp(x_copy + delta, min=0, max=1).detach()
        loss20_history.append(loss)
        x_adv = x_tmp
    for i in range(num_iter):
        if i ==0:
            accuracy_list.append(1)
        if i > 0:
            numerator = loss_history[i] - loss_history[i-1]
            denominator = loss20_history[i-1] - loss_history[i-1]
            AS = numerator/denominator
            float_AS = float(AS)
            print(float_AS)
            accuracy_list.append(float_AS)
    if len(accuracy_list) == num_iter*50:
        for j in range(len(accuracy_list)):
            if j%20 == 0:
                avg_accuracy[0] +=  accuracy_list[j]/50
            if j%20 == 1:
                avg_accuracy[1] +=  accuracy_list[j]/50
            if j%20 == 2:
                avg_accuracy[2] +=  accuracy_list[j]/50
            if j%20 == 3:
                avg_accuracy[3] +=  accuracy_list[j]/50
            if j%20 == 4:
                avg_accuracy[4] +=  accuracy_list[j]/50
            if j%20 == 5:
                avg_accuracy[5] +=  accuracy_list[j]/50
            if j%20 == 6:
                avg_accuracy[6] +=  accuracy_list[j]/50
            if j%20 == 7:
                avg_accuracy[7] +=  accuracy_list[j]/50
            if j%20 == 8:
                avg_accuracy[8] +=  accuracy_list[j]/50
            if j%20 == 9:
                avg_accuracy[9] +=  accuracy_list[j]/50
            if j%20 == 10:
                avg_accuracy[10] +=  accuracy_list[j]/50
            if j%20 == 11:
                avg_accuracy[11] +=  accuracy_list[j]/50
            if j%20 == 12:
                avg_accuracy[12] +=  accuracy_list[j]/50
            if j%20 == 13:
                avg_accuracy[13] +=  accuracy_list[j]/50
            if j%20 == 14:
                avg_accuracy[14] +=  accuracy_list[j]/50
            if j%20 == 15:
                avg_accuracy[15] +=  accuracy_list[j]/50
            if j%20 == 16:
                avg_accuracy[16] +=  accuracy_list[j]/50
            if j%20 == 17:
                avg_accuracy[17] +=  accuracy_list[j]/50
            if j%20 == 18:
                avg_accuracy[18] +=  accuracy_list[j]/50
            if j%20 == 19:
                avg_accuracy[19] +=  accuracy_list[j]/50
    if len(accuracy_list) == num_iter*50:
        for i in range(num_iter):
            print(avg_accuracy[i])
        plt.plot(avg_accuracy, marker='o')
        plt.title("line_chart")
        plt.xlabel("interations")
        plt.ylabel("AS")
        plt.savefig("line_chart.png")
    return x_adv

# Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks (ICLR 2020)'(https://arxiv.org/abs/1908.06281)
def nifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_new = x_adv + alpha*decay*momentum
        x_new.requires_grad = True 
        loss = loss_fn(model(x_new), y)
        loss.backward() 
        grad = x_new.grad.detach() 
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)      
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def nifgsm20(model, x, y, loss_fn, epsilon=epsilon/3, alpha=alpha/6, beta=alpha/30,num_iter=20, num=5,decay=0.7):
    loss_history = []
    loss20_history = []
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_new = x_adv + alpha*decay*momentum
        x_new.requires_grad = True
        x_copy = x_adv
        loss = loss_fn(model(x_new), y)
        loss_history.append(loss)
        loss.backward()
        grad = x_new.grad.detach()
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        x_tmp = x_adv
        x_adv = x_copy
        for j in range(num):
            x_adv = x_adv.detach().clone()
            x_new = x_adv + alpha * decay * momentum
            x_new.requires_grad = True
            loss = loss_fn(model(x_new), y)
            loss.backward()
            grad = x_new.grad.detach()
            x_adv = x_adv +  beta * grad.sign()
            delta = torch.clamp(x_adv - x_copy, min=-alpha, max=alpha)
            x_adv = torch.clamp(x_copy + delta, min=0, max=1).detach()
        loss20_history.append(loss)
        x_adv = x_tmp
    for i in range(num_iter):
        if i ==0:
            accuracy_list.append(1)
        if i > 0:
            numerator = loss_history[i] - loss_history[i-1]
            denominator = loss20_history[i-1] - loss_history[i-1]
            AS = numerator/denominator
            float_AS = float(AS)
            print(float_AS)
            accuracy_list.append(float_AS)
    if len(accuracy_list) == num_iter*50:
        for j in range(len(accuracy_list)):
            if j%20 == 0:
                avg_accuracy[0] +=  accuracy_list[j]/50
            if j%20 == 1:
                avg_accuracy[1] +=  accuracy_list[j]/50
            if j%20 == 2:
                avg_accuracy[2] +=  accuracy_list[j]/50
            if j%20 == 3:
                avg_accuracy[3] +=  accuracy_list[j]/50
            if j%20 == 4:
                avg_accuracy[4] +=  accuracy_list[j]/50
            if j%20 == 5:
                avg_accuracy[5] +=  accuracy_list[j]/50
            if j%20 == 6:
                avg_accuracy[6] +=  accuracy_list[j]/50
            if j%20 == 7:
                avg_accuracy[7] +=  accuracy_list[j]/50
            if j%20 == 8:
                avg_accuracy[8] +=  accuracy_list[j]/50
            if j%20 == 9:
                avg_accuracy[9] +=  accuracy_list[j]/50
            if j%20 == 10:
                avg_accuracy[10] +=  accuracy_list[j]/50
            if j%20 == 11:
                avg_accuracy[11] +=  accuracy_list[j]/50
            if j%20 == 12:
                avg_accuracy[12] +=  accuracy_list[j]/50
            if j%20 == 13:
                avg_accuracy[13] +=  accuracy_list[j]/50
            if j%20 == 14:
                avg_accuracy[14] +=  accuracy_list[j]/50
            if j%20 == 15:
                avg_accuracy[15] +=  accuracy_list[j]/50
            if j%20 == 16:
                avg_accuracy[16] +=  accuracy_list[j]/50
            if j%20 == 17:
                avg_accuracy[17] +=  accuracy_list[j]/50
            if j%20 == 18:
                avg_accuracy[18] +=  accuracy_list[j]/50
            if j%20 == 19:
                avg_accuracy[19] +=  accuracy_list[j]/50
    if len(accuracy_list) == num_iter*50:
        for i in range(num_iter):
            print(avg_accuracy[i])
        plt.plot(avg_accuracy, marker='o')
        plt.title("line_chart")
        plt.xlabel("interations")
        plt.ylabel("AS")
        plt.savefig("line_chart.png")
    return x_adv







def vmi_fgsm(model, x, y, loss_fn, alpha=alpha, num_iter=10, decay=1, N = 5 ,beta = 3/2 ,epsilon=epsilon):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    v = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()  
        x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad
        out = model(x_adv)  
        loss = loss_fn(out, y)                
        loss.backward() 
        
        grad1 = x_adv.grad.detach()
        grad = decay * momentum + (grad1+v)/torch.mean(torch.abs(grad1+v), dim=(1,2,3), keepdim=True)
        momentum = grad    
        # Calculate Gradient Variance
        GV_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            neighbor_images = x_adv.detach() + torch.randn_like(x).uniform_(-epsilon*beta, epsilon*beta) #ji 的把0.15改成
            neighbor_images.requires_grad = True
            out = model(neighbor_images)
            cost = loss_fn(out, y)
            cost.backward()
            grad2 = neighbor_images.grad.detach()
            GV_grad += grad2
        # obtaining the gradient variance
        v = GV_grad / N - grad1 
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        # x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv


# AITM 2022/aaai
def get_alpha(T,t_,beta1=0.9, beta2=0.99):
    res = 0
    for t in range(T):
        res += (1-beta1**(t+1))/math.sqrt(1-beta2**(t+1))
    return epsilon/res * (1-beta1**(t_+1))/math.sqrt(1-beta2**(t_+1))
    
def aitm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10,mu1=1.5,mu2=1.9, decay=1.0, beta1=0.9, beta2=0.99,r = 1.3, adam_eps=1e-8):
    x_adv = x
    #2022 AAAI Making Adversarial Examples More Transferable and Indistinguishable
    # initialize momentum tensor and second moment tensor
    momentum = torch.zeros_like(x).detach().cuda()
    v = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True   
        out = model(x_adv)  
        loss = loss_fn(out, y)                
        loss.backward()  
        grad = x_adv.grad.detach()  
        momentum = momentum + mu1*grad
        v = v + mu2 * (grad ** 2)
        m_hat =  r*( momentum/ (torch.sqrt(v) + adam_eps))
        alpha = get_alpha(num_iter,i)
        x_adv = x_adv + alpha * m_hat.tanh()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv


#NeurIPS'2023 PGN 分超级高(zeta =3 时) 该论文必须好好的借鉴 非常不错 为什么高 
def pgn(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=3.0,aa=0.5,N=20):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True) 
            out = model(x_near)  
            loss = loss_fn(out, y)                
            loss.backward() 
            g1 = x_near.grad.detach() 
            x_star = x_near.detach() + alpha * (-g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(nes_x)
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            avg_grad += (1-aa)*g1 + aa*g2
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        #x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv

def get_cosine_similarity(cur_grad, sam_grad):
    cur_grad = cur_grad.view(cur_grad.size(0), -1)
    sam_grad = sam_grad.view(sam_grad.size(0), -1)
    cos_sim = torch.sum(cur_grad * sam_grad, dim=1) / (torch.sqrt(torch.sum(cur_grad ** 2, dim=1)) * torch.sqrt(torch.sum(sam_grad ** 2, dim=1)))
    cos_sim = cos_sim.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return cos_sim

def pgn1(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10,beta=1.5,aa=0.5,N=5,lmbda=0.999):#原文N为20 yuan zeta = 3
    x_adv = x
    # initialize momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(x_near)
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * (-g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(nes_x)
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            avg_grad += (1-aa)*g1 + aa*g2
        grad = avg_grad / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = lmbda * momentum + (1-lmbda)*grad
        momentum = grad
        sigma = get_cosine_similarity(grad, avg_grad)
        d = sigma * momentum - avg_grad
        x_adv = x_adv + alpha * torch.sign(d)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        #x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv

def pgn2(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=3.0,aa=0.0,bb=0.0,N=20):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        sum_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(x_near)
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(nes_x)
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            x_sun = x_near.detach() - alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            on_x = x_sun.detach()
            on_x = Variable(on_x, requires_grad=True)
            out = model(on_x)
            loss = loss_fn(out, y)
            loss.backward()
            g3 = on_x.grad.detach()
            sum_grad += (1-aa-bb)*g1 + aa*g2 + bb*g3
        avg_grad = sum_grad/N
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def cap(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=3.0,aa=0.2,bb=0.5,N=20):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        sum_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(x_near)
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(nes_x)
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            x_sun = x_near.detach() - alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            on_x = x_sun.detach()
            on_x = Variable(on_x, requires_grad=True)
            out = model(on_x)
            loss = loss_fn(out, y)
            loss.backward()
            g3 = on_x.grad.detach()
            sum_grad += (1-aa-bb)*g1 + aa*g2 + bb*g3
        avg_grad = sum_grad/N
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv


def cosine_similarity(g1, g2):
    dot_product = torch.sum(g1 * g2, dim=(1, 2, 3))  # 计算点积
    norm_g1 = torch.norm(g1, p=2, dim=(1, 2, 3))  # 计算L2范数
    norm_g2 = torch.norm(g2, p=2, dim=(1, 2, 3))  # 计算L2范数
    return dot_product / (norm_g1 * norm_g2)

def pcr(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0, beta=3, aa=0.5, bb=0.4, N=20, P=5, S = 10, sigma=1):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    grad_history = []
    for i in range(P):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(x_near)
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * (-g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(nes_x)
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            x_sun = x_near.detach() - alpha * (-g1) / torch.abs(g1).mean([1, 2, 3], keepdim=True)
            on_x = x_sun.detach()
            on_x = Variable(on_x, requires_grad=True)
            out = model(on_x)
            loss = loss_fn(out, y)
            loss.backward()
            g3 = on_x.grad.detach()
            avg_grad += aa*g1 + bb*g2 + (1-aa-bb)*g3
        grad_norm = torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = avg_grad / grad_norm
        grad = decay * momentum + grad
        momentum = grad
        if len(grad_history) > 0:
            grad_similarity = get_cosine_similarity(grad,grad_history[-1])
            sigma = grad_similarity.mean().item()
            print(sigma)
        x_adv = x_adv + (S/sigma) * alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        grad_history.append(grad)
        if len(grad_history) > N:
            grad_history.pop(0)
    x_adv = x
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        avg_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(x_near)
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * (-g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(nes_x)
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            x_sun = x_near.detach() - alpha * (-g1) / torch.abs(g1).mean([1, 2, 3], keepdim=True)
            on_x = x_sun.detach()
            on_x = Variable(on_x, requires_grad=True)
            out = model(on_x)
            loss = loss_fn(out, y)
            loss.backward()
            g3 = on_x.grad.detach()
            avg_grad += aa*g1 + bb*g2 + (1-aa-bb)*g3
        grad_norm = torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = avg_grad / grad_norm
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def pgn20(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=3.0,aa=0.5, pha=alpha/5, num=5):
    loss_history = []
    loss20_history = []
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_copy = x_adv
        avg_grad = torch.zeros_like(x).detach().cuda()
        x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
        x_near = Variable(x_near, requires_grad = True)
        out = model(x_near)
        loss = loss_fn(out, y)
        loss.backward()
        g1 = x_near.grad.detach()
        x_star = x_near.detach() + alpha * (-g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)
        nes_x = x_star.detach()
        nes_x = Variable(nes_x, requires_grad = True)
        out = model(nes_x)
        loss = loss_fn(out, y)
        loss_history.append(loss)
        loss.backward()
        g2 = nes_x.grad.detach()
        avg_grad += (1-aa)*g1 + aa*g2
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        x_tmp = x_adv
        x_adv = x_copy
        avg_grad = torch.zeros_like(x).detach().cuda()
        for j in range(num):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon * beta, epsilon * beta)
            x_near = Variable(x_near, requires_grad=True)
            out = model(x_near)
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * (-g1) / torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad=True)
            out = model(nes_x)
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            avg_grad += (1 - aa) * g1 + aa * g2
            grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
            grad = decay * momentum + grad
            momentum = grad
            x_adv = x_adv + pha * torch.sign(grad)
            delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        loss20_history.append(loss)
        x_adv = x_tmp
    for i in range(num_iter):
        if i > 0:
            numerator = loss_history[i] - loss_history[i - 1]
            denominator = loss20_history[i - 1] - loss_history[i - 1]
            AS = numerator / denominator
            float_AS = float(AS)
            print(float_AS)
            accuracy_list.append(float_AS)
    if len(accuracy_list) == (num_iter - 1) * 50:
        for j in range(len(accuracy_list)):
            if j % 19 == 0:
                avg_accuracy[0] += accuracy_list[j] / 50
            if j % 19 == 1:
                avg_accuracy[1] += accuracy_list[j] / 50
            if j % 19 == 2:
                avg_accuracy[2] += accuracy_list[j] / 50
            if j % 19 == 3:
                avg_accuracy[3] += accuracy_list[j] / 50
            if j % 19 == 4:
                avg_accuracy[4] += accuracy_list[j] / 50
            if j % 19 == 5:
                avg_accuracy[5] += accuracy_list[j] / 50
            if j % 19 == 6:
                avg_accuracy[6] += accuracy_list[j] / 50
            if j % 19 == 7:
                avg_accuracy[7] += accuracy_list[j] / 50
            if j % 19 == 8:
                avg_accuracy[8] += accuracy_list[j] / 50
            if j % 19 == 9:
                avg_accuracy[9] += accuracy_list[j] / 50
            if j % 19 == 10:
                avg_accuracy[10] += accuracy_list[j] / 50
            if j % 19 == 11:
                avg_accuracy[11] += accuracy_list[j] / 50
            if j % 19 == 12:
                avg_accuracy[12] += accuracy_list[j] / 50
            if j % 19 == 13:
                avg_accuracy[13] += accuracy_list[j] / 50
            if j % 19 == 14:
                avg_accuracy[14] += accuracy_list[j] / 50
            if j % 19 == 15:
                avg_accuracy[15] += accuracy_list[j] / 50
            if j % 19 == 16:
                avg_accuracy[16] += accuracy_list[j] / 50
            if j % 19 == 17:
                avg_accuracy[17] += accuracy_list[j] / 50
            if j % 19 == 18:
                avg_accuracy[18] += accuracy_list[j] / 50
    if len(accuracy_list) == (num_iter - 1) * 50:
        for i in range(num_iter - 1):
            print(avg_accuracy[i])
        plt.plot(avg_accuracy, marker='o')
        plt.title("line_chart")
        plt.xlabel("interations")
        plt.ylabel("AS")
        plt.savefig("line_chart.png")

        #x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv

def gmifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,P = 5,S = 10):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(P):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        grad = x_adv.grad.detach()
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        x_adv = x_adv + S * alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    x_adv = x
    for j in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        grad = x_adv.grad.detach()
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv


def gvmi_fgsm(model, x, y, loss_fn, alpha=alpha, num_iter=10, decay=1, N=20, beta=3/2, epsilon=epsilon, P = 5, S = 10):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    v = torch.zeros_like(x).detach().cuda()
    for i in range(P):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True  # need to obtain gradient of x_adv, thus set required grad
        out = model(x_adv)
        loss = loss_fn(out, y)
        loss.backward()

        grad1 = x_adv.grad.detach()
        grad = decay * momentum + (grad1 + v) / torch.mean(torch.abs(grad1 + v), dim=(1, 2, 3), keepdim=True)
        momentum = grad
        # Calculate Gradient Variance
        GV_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            neighbor_images = x_adv.detach() + torch.randn_like(x).uniform_(-epsilon * beta,
                                                                            epsilon * beta)  # ji 的把0.15改成
            neighbor_images.requires_grad = True
            out = model(neighbor_images)
            cost = loss_fn(out, y)
            cost.backward()
            grad2 = neighbor_images.grad.detach()
            GV_grad += grad2
        # obtaining the gradient variance
        v = GV_grad / N - grad1
        x_adv = x_adv + S * alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        # x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    x_adv = x
    for j in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True  # need to obtain gradient of x_adv, thus set required grad
        out = model(x_adv)
        loss = loss_fn(out, y)
        loss.backward()

        grad1 = x_adv.grad.detach()
        grad = decay * momentum + (grad1 + v) / torch.mean(torch.abs(grad1 + v), dim=(1, 2, 3), keepdim=True)
        momentum = grad
        # Calculate Gradient Variance
        GV_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            neighbor_images = x_adv.detach() + torch.randn_like(x).uniform_(-epsilon * beta,
                                                                            epsilon * beta)  # ji 的把0.15改成
            neighbor_images.requires_grad = True
            out = model(neighbor_images)
            cost = loss_fn(out, y)
            cost.backward()
            grad2 = neighbor_images.grad.detach()
            GV_grad += grad2
        # obtaining the gradient variance
        v = GV_grad / N - grad1
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        # x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv


#  2023/ICCV 论文GRA
def get_decay_indicator(M, x, cur_noise, last_noise, eta,):
    
    if isinstance(last_noise, int):
        last_noise = torch.full(cur_noise.shape, last_noise)
    else:
        last_noise = last_noise
    if torch.cuda.is_available():
        last_noise = last_noise.cuda()
    last = last_noise.sign()
    cur = cur_noise.sign()
    eq_m = (last == cur).float()
    di_m = torch.ones_like(x) - eq_m
    M = M * (eq_m + di_m * eta)
    return M    
    
def gra(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha,N=5,num_iter=10, decay=1.0, beta= 3/2,eta=0.94):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    
    # Initialize the decay indicator
    M = torch.full_like(x, 1 / eta)    
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()   
        x_adv.requires_grad = True
        out = model(x_adv)
        cost = loss_fn(out, y)
        cost.backward()
        grad1 = x_adv.grad.detach()
        GV_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            
            neighbor_images = x_adv.detach() + torch.randn_like(x).uniform_(-epsilon*beta, epsilon*beta) #ji 的把0.15改成
            neighbor_images.requires_grad = True
            out = model(neighbor_images)
            cost = loss_fn(out, y)
            cost.backward()
            grad_n = neighbor_images.grad.detach()
            GV_grad += grad_n
        grad_n = GV_grad / N    
        cossim = get_cosine_similarity(grad1,grad_n)
        c_grad = cossim*grad1 + (1-cossim)*grad_n
        last_momentum = momentum
        grad = decay*momentum +(c_grad)/torch.mean(torch.abs(c_grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        
        M = get_decay_indicator(M,x, momentum, last_momentum, eta)
        x_adv = x_adv + alpha*M*grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv



# ICCV'2023   Structure Invariant Transformation for better Adversarial Transferability  SIA
def sia(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha,num_copies=5,num_block=3, num_iter=10, decay=1.0):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    def vertical_shift( x):
        _, _, w, _ = x.shape
        step = np.random.randint(low = 0, high=w, dtype=np.int32)
        return x.roll(step, dims=2)
    def horizontal_shift( x):
        _, _, _, h = x.shape
        step = np.random.randint(low = 0, high=h, dtype=np.int32)
        return x.roll(step, dims=3)
    def vertical_flip( x):
        return x.flip(dims=(2,))
    def horizontal_flip( x):
        return x.flip(dims=(3,))
    def rotate180( x):
        return x.rot90(k=2, dims=(2,3))
    def scale( x):
        return torch.rand(1)[0] * x
    def resize( x):
        """
        Resize the input
        """
        _, _, w, h = x.shape
        scale_factor = 0.8
        new_h = int(h * scale_factor)+1
        new_w = int(w * scale_factor)+1
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(w, h), mode='bilinear', align_corners=False).clamp(0, 1)
        return x
    
    def dct( x):
        """
        Discrete Fourier Transform
        """
        dctx = torch.fft.fftn(x, dim=(-2, -1))
        
        #dctx = dct.dct_2d(x) #torch.fft.fftn(x, dim=(-2, -1))
        _, _, w, h = dctx.shape
        low_ratio = 0.4
        low_w = int(w * low_ratio)
        low_h = int(h * low_ratio)
        # dctx[:, :, -low_w:, -low_h:] = 0
        dctx[:, :, -low_w:,:] = 0
        dctx[:, :, :, -low_h:] = 0
        dctx = dctx # * mask.reshape(1, 1, w, h)
        
        idctx = torch.fft.ifftn(x, dim=(-2, -1))
        #idctx = dct.idct_2d(dctx)
        return idctx
    def add_noise( x):
        return torch.clip(x + torch.zeros_like(x).uniform_(-16/255,16/255), 0, 1)
    def gkern( kernel_size=3, nsig=3):
        x = np.linspace(-nsig, nsig, kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return torch.from_numpy(stack_kernel.astype(np.float32)).cuda()

    def drop_out( x):
        
        return F.dropout2d(x, p=0.1, training=True) 
    def blocktransform( x, choice=-1):
        _, _, w, h = x.shape
        y_axis = [0,] + np.random.choice(list(range(1, h)), num_block-1, replace=False).tolist() + [h,]
        x_axis = [0,] + np.random.choice(list(range(1, w)), num_block-1, replace=False).tolist() + [w,]
        y_axis.sort()
        x_axis.sort()
        
        x_copy = x.clone()
        for i, idx_x in enumerate(x_axis[1:]):
            for j, idx_y in enumerate(y_axis[1:]):
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(op), dtype=np.int32)
                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = op[chosen](x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y])

        return x_copy
    def transform( x):
        """
        Scale the input for BlockShuffle
        """
        return torch.cat([blocktransform(x) for _ in range(num_copies)])

    for i in range(num_iter):
        x_adv = x_adv.detach().clone()      
        x_adv.requires_grad = True 
       
        op = [resize,vertical_shift,horizontal_shift,vertical_flip,horizontal_flip,rotate180,scale,add_noise,drop_out,dct]#add_noise,drop_out
        x_other = transform(x_adv)
        out = model(x_other)    
        #loss = loss_fn(out, y)        
        loss = loss_fn(out, y.repeat(num_copies))       
        loss.backward() 
        grad = x_adv.grad.detach() 
        grad = decay * momentum +  grad / (grad.abs().sum() + 1e-8)       
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
        #x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv















#Y  cvpr2024
def get_length(length):
    num_block=3
    rand = np.random.uniform(size=num_block)
    rand_norm = np.round(rand/rand.sum()*length).astype(np.int32)
    rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
    return tuple(rand_norm)

def shuffle_single_dim(x, dim):
    lengths = get_length(x.size(dim))
    x_strips = list(x.split(lengths, dim=dim))
    random.shuffle(x_strips)
    return x_strips


def shuffle1(x):
    dims = [2,3]
    random.shuffle(dims)
    x_strips = shuffle_single_dim(x, dims[0])
    return torch.cat([torch.cat(shuffle_single_dim(x_strip, dim=dims[1]), dim=dims[1]) for x_strip in x_strips], dim=dims[0])



def bsr(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, M=20,num_iter=10, num_block=3,decay=1.0):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        grad_m = 0
        for n in range(M):
            x_new=shuffle1(x_adv) 
            out = model(x_new)  
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

def si_fgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, M=5,num_iter=10,decay=1.0):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().cuda()
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        
        grad_m = 0
        for n in range(M):
            x_new= x_adv/(2**n) 
            x_new.requires_grad = True
            out = model(x_new)  
            loss = loss_fn(out, y)
            loss.backward()
            grad_m += x_new.grad.detach()
        grad = grad_m / M
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)      
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def DI(x, resize_rate=1.15, diversity_prob=0.5):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    ret = padded if torch.rand(1) < diversity_prob else x
    return ret

def dicap(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=3.0,aa=0.2,bb=0.5,N=20):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        sum_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(DI(x_near))
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(DI(nes_x))
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            x_sun = x_near.detach() - alpha * g1 / torch.abs(g1).mean([1, 2, 3], keepdim=True)
            on_x = x_sun.detach()
            on_x = Variable(on_x, requires_grad=True)
            out = model(DI(on_x))
            loss = loss_fn(out, y)
            loss.backward()
            g3 = on_x.grad.detach()
            sum_grad += (1-aa-bb)*g1 + aa*g2 + bb*g3
        avg_grad = sum_grad / N
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def gkern(kernlen=5, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def TI(grad_in, kernel_size=5):
    kernel = gkern(kernel_size, 3).astype(np.float32)
    gaussian_kernel = np.stack([kernel, kernel, kernel])
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
    gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()
    #conv2d(grad2, TI_kernel(), bias=None, stride=1, padding=(2,2), groups=3)
    grad_out = F.conv2d(grad_in, gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3) #TI
    return grad_out

def ticap(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=3.0,aa=0.2,bb=0.5,N=20):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        sum_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(x_near)
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(nes_x)
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            x_sun = x_near.detach() - alpha * g1 / torch.abs(g1).mean([1, 2, 3], keepdim=True)
            on_x = x_sun.detach()
            on_x = Variable(on_x, requires_grad=True)
            out = model(on_x)
            loss = loss_fn(out, y)
            loss.backward()
            g3 = on_x.grad.detach()
            sum_grad += (1-aa-bb)*g1 + aa*g2 + bb*g3
        avg_grad = sum_grad / N
        avg_grad = TI(avg_grad)
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def sicap(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=3.0,aa=0.2,bb=0.5,N=20):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        sum_grad = torch.zeros_like(x).detach().cuda()
        for m in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(x_near/(2**m))
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(nes_x/(2**m))
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            x_sun = x_near.detach() - alpha * g1 / torch.abs(g1).mean([1, 2, 3], keepdim=True)
            on_x = x_sun.detach()
            on_x = Variable(on_x, requires_grad=True)
            out = model(on_x/(2**m))
            loss = loss_fn(out, y)
            loss.backward()
            g3 = on_x.grad.detach()
            sum_grad += (1-aa-bb)*g1 + aa*g2 + bb*g3
        avg_grad = sum_grad / N
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def mix(x_adv):   # 跟dmi的用法一样  直接放到模型的输出哪里  model(mix(xadv))
    img_other = x_adv[torch.randperm(x_adv.shape[0])].view(x_adv.size())
    xx= x_adv + 0.2 * img_other
    return xx

def adcap(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=3.0,aa=0.2,bb=0.5,N=20):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        sum_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(mix(x_near))
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(mix(nes_x))
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            x_sun = x_near.detach() - alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            on_x = x_sun.detach()
            on_x = Variable(on_x, requires_grad=True)
            out = model(mix(on_x))
            loss = loss_fn(out, y)
            loss.backward()
            g3 = on_x.grad.detach()
            sum_grad += (1-aa-bb)*g1 + aa*g2 + bb*g3
        avg_grad = sum_grad/N
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def ssa(x,rho=0.5):
    _,_,_,image_width = x.size()
    gauss = torch.randn(x.size()[0], 3, image_width, image_width) * epsilon
    gauss = gauss.cuda()
    x_dct = dct_2d(x + gauss).cuda()
    mask = (torch.rand_like(x) * 2 * rho + 1 - rho).cuda()
    x_idct = idct_2d(x_dct * mask)
    return x_idct

def ssacap(model, x, y, loss_fn,epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0,beta=3.0,aa=0.2,bb=0.5,N=20):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        sum_grad = torch.zeros_like(x).detach().cuda()
        for _ in range(N):
            x_near = x_adv + torch.rand_like(x).uniform_(-epsilon*beta, epsilon*beta)
            x_near = Variable(x_near, requires_grad = True)
            out = model(ssa(x_near))
            loss = loss_fn(out, y)
            loss.backward()
            g1 = x_near.grad.detach()
            x_star = x_near.detach() + alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            nes_x = x_star.detach()
            nes_x = Variable(nes_x, requires_grad = True)
            out = model(nes_x)
            loss = loss_fn(out, y)
            loss.backward()
            g2 = nes_x.grad.detach()
            x_sun = x_near.detach() - alpha * g1/torch.abs(g1).mean([1, 2, 3], keepdim=True)
            on_x = x_sun.detach()
            on_x = Variable(on_x, requires_grad=True)
            out = model(on_x)
            loss = loss_fn(out, y)
            loss.backward()
            g3 = on_x.grad.detach()
            sum_grad += (1-aa-bb)*g1 + aa*g2 + bb*g3
        avg_grad = sum_grad/N
        grad = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
        grad = decay * momentum + grad
        momentum = grad
        x_adv = x_adv + alpha * torch.sign(grad)
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def dim(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(DI(x_adv)), y)
        loss.backward()
        grad = x_adv.grad.detach()
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def tim(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        grad = x_adv.grad.detach()
        grad = TI(grad)
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def sim(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for m in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(x_adv/(2**m)), y)
        loss.backward()
        grad = x_adv.grad.detach()
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def admix(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(mix(x_adv)), y)
        loss.backward()
        grad = x_adv.grad.detach()
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv

def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.fft.fft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1).real.view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    V = Vc.real * W_r - Vc.imag * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
    v = torch.fft.ifft(tmp)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape).real


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)

def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)

def SSA(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=10, decay=1.0):
    x_adv = x
    momentum = torch.zeros_like(x).detach().cuda()
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(ssa(x_adv)), y)
        loss.backward()
        grad = x_adv.grad.detach()
        grad = decay * momentum +  grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, min=0, max=1).detach()
    return x_adv
