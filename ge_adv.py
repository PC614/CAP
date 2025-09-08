"""Implementation of sample attack."""
import os
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import ToTensor, ToPILImage, transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
import pretrainedmodels
from os.path import join, dirname
from styleaug import StyleAugmentor
import torchvision
from attack_methods import *
import random
import timm
from decowa import *
parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./outputs/', help='Output directory with adversarial images.')

parser.add_argument("--batch_size", type=int, default=20, help="How many images process at one time.")

#parser.add_argument("--N", type=int, default=20, help="The number of Spectrum Transformations (Sampling Number)")


parser.add_argument('--attack', default='fgsm', type=str, help='fgsm, ifgsm, mifgsm, vmi_fgsm, ssi,ssmi, si_fgsm,smi_fgsm,admix,mmi_fgsm')

# 攻击的模型 densenet121、inception_v3、resnet50、swin-t、vit -- "Ensemble_Model"---- tf2torch_ens3_adv_inc_v3 、tf2torch_ens4_adv_inc_v3 、tf2torch_ens_adv_inc_res_v2

parser.add_argument('--name', default='inception_v3', type=str, help='model')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

accuracy_list = []
avg_accuracy = [0]*99

## -- 代理模型 用'inception_v3、resnet50 densenet121 偶尔也可以用vit   对抗防御模型本文不做代理模型


def seed_torch(seed):
    """Set a random seed to ensure that the results are reproducible"""  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False





def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)


def get_model(net_name):
    """Load converted model"""
    #model_path = os.path.join(model_dir, net_name + '.npy')

 
    if net_name == 'inception_v3':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                torchvision.models.inception_v3(pretrained=True).eval().cuda())                         
        
    elif net_name == 'densenet121':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                torchvision.models.densenet121(pretrained=True).eval().cuda())
        
    elif net_name == 'resnet50':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                torchvision.models.resnet50(pretrained=True).eval().cuda())                
    elif net_name == 'swin-t':
        
        model1 = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        mean1 = model1.default_cfg['mean']
        std1 = model1.default_cfg['std'] 
        model = torch.nn.Sequential(Normalize(list(mean1), list(std1)),
                                model1.eval().cuda())        
    elif net_name == 'vit':
        
        model1 = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        mean1 = model1.default_cfg['mean']
        std1 = model1.default_cfg['std'] 
        model = torch.nn.Sequential(Normalize(list(mean1), list(std1)),model1.eval().cuda())        
   
    else:
        print('Wrong model name!')

    return model


def generate_attack(model,data_loader, attack):
    
    
    for images, images_ID,  gt_cpu in tqdm(data_loader):
        y = gt_cpu.cuda()
        x = images.cuda()
        x_adv = attack(model,x, y,loss_fn)
        
        adv_img_np5 = x_adv.cpu().data.numpy()
        adv_img_np5 = np.transpose(adv_img_np5, (0, 2, 3, 1)) * 255
        
        output_dir = "./adv_data/adv_%s_%s/"%(opt.name,opt.attack)
        
        save_image(adv_img_np5, images_ID, output_dir)
        
        

class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super(EnsembleModel, self).__init__()
        self.device = next(models[0].parameters()).device
        for model in models:
            model.to(self.device)
        self.models = models
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        if self.mode == 'mean':
            outputs = torch.mean(outputs, dim=0)
            return outputs
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError


            
            
def main():
    
    if opt.name ==  'inception_v3':
        img_size = 299
    else:
        img_size = 224

    transforms = T.Compose([T.Resize(img_size), T.ToTensor()])    
    
    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)


    attack = eval(opt.attack) # 将双引号去掉
    
    if opt.name == "Ensemble_Model":
        Ens_m = ['resnet50','densenet121','vit']
        model = EnsembleModel([get_model(name) for name in Ens_m])
    else:
    
        model_name = opt.name
        model = get_model(model_name)   
    
    
    

    
    generate_attack(model,data_loader, attack)


if __name__ == '__main__':
    seed_torch(0)
    main()

    
    

  









    
