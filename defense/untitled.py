import argparse
import os
import torch
import datetime
import torch.backends.cudnn as cudnn
from torchvision.models.resnet import resnet50
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
from torchvision import transforms
import torch.nn as nn

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):

        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        self.base_classifier.eval()
        counts_selection = self._sample_noise(x, n0, batch_size)
        cAHat = counts_selection.argmax().item()
        counts_estimation = self._sample_noise(x, n, batch_size)
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:

        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:

        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
 
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]


class NormalizeLayer(torch.nn.Module):


    def __init__(self, means, sds):

        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000


def get_architecture(arch: str, dataset: str) -> torch.nn.Module:

    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).cuda()
        cudnn.benchmark = True
    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)


class smooth_defense(nn.Module):
    def __init__(self,
                 skip: int = 100,
                 N: int = 100,
                 batch_size: int = 400,
                 path: str = './models/defense/rs_imagenet/resnet50/noise_0.25/checkpoint.pth.tar',
                 dataset: str = 'imagenet',
                 sigma: float = 0.25,
                 alpha: float = 0.001):
        super(smooth_defense, self).__init__()
        self.skip = skip
        self.N = N
        self.batch_size = batch_size
        self.path = path
        self.dataset = dataset
        self.sigma = sigma
        self.alpha = alpha

        self.checkpoint = torch.load(self.path)
        self.base_classifier = get_architecture(self.checkpoint["arch"], self.dataset)
        self.base_classifier.load_state_dict(self.checkpoint['state_dict'])

        self.smooth_classifier = Smooth(self.base_classifier, get_num_classes(self.dataset), self.sigma)

    def trn(self, img):
        img = transforms.ToPILImage()(img)
        trn_img = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        return trn_img(img)


    def forward(self, img, gt):
        sum = 0
        for n in range(img.shape[0]):
            signle_img = self.trn(img[n]).cuda()
            prediction = self.smooth_classifier.predict(signle_img, self.N, self.alpha, self.batch_size)
            incorrect = int(prediction != gt[n].item())
            sum += incorrect
        return sum
    
    

