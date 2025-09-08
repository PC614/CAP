"""Implementation of evaluate attack result."""
import os
import torch
import torchvision
from torch.autograd import Variable as V
from torch import nn
from torchvision import transforms as T
from Normalize import Normalize, TfNormalize
from loader import ImageNet
from torch.utils.data import DataLoader
import pretrainedmodels
import timm
import argparse
from attack_methods import *
from decowa import *

from PIL import Image
from defense.RP import *
from defense.NRP import *
from defense.JPEG import *
from defense.Bit_red import *
from defense.HGD.HGD import *
from defense.FD import *

from defense.RS import *
parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./outputs/', help='Output directory with adversarial images.')
parser.add_argument('--mean', type=float, default=np.array([0.485, 0.456, 0.406]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.229, 0.224, 0.225]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=10, help="How many images process at one time.")


parser.add_argument('--attack', default='fgsm', type=str, help='fgsm, ifgsm, mifgsm, vmi_fgsm, ssi,ssmi, si_fgsm,smi_fgsm,admix,mmi_fgsm')

# 攻击的模型 densenet121、inception_v3、resnet50、swin-t、vit --"Ensemble_Model"---- tf2torch_ens3_adv_inc_v3 、tf2torch_ens4_adv_inc_v3 、tf2torch_ens_adv_inc_res_v2

parser.add_argument('--name', default='inception_v3', type=str, help='model')
opt = parser.parse_args()







input_csv = './dataset/images.csv'
input_dir = './dataset/images'
adv_dir = "./adv_data/adv_%s_%s"%(opt.name,opt.attack)
output_dir = './output_defense/'
file_name = '%s_%s_result.txt'%(opt.name,opt.attack)

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_path = os.path.join(output_dir, file_name)



    




def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

 
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
        model = torch.nn.Sequential(Normalize(list(mean1), list(std1)),
                                model1.eval().cuda())        

        
    elif net_name == 'inc_res_v2':
        model = torch.nn.Sequential(Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                pretrainedmodels.inceptionresnetv2(num_classes=1000, pretrained='imagenet').eval().cuda())
        
        
        
    elif net_name == 'tf2torch_adv_inception_v3':
        from torch_nets import tf_adv_inception_v3
        net = tf_adv_inception_v3
        model = nn.Sequential( 
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),)
    elif net_name == 'tf2torch_ens3_adv_inc_v3':
        from torch_nets import tf_ens3_adv_inc_v3
        net = tf_ens3_adv_inc_v3
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),)
    elif net_name == 'tf2torch_ens4_adv_inc_v3':
        from torch_nets import tf_ens4_adv_inc_v3
        net = tf_ens4_adv_inc_v3
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),)
    elif net_name == 'tf2torch_ens_adv_inc_res_v2':
        from torch_nets import tf_ens_adv_inc_res_v2
        net = tf_ens_adv_inc_res_v2
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),)
    else:
        print('Wrong model name!')

    return model












def verify(model_name, path):
    
    model = get_model(model_name, path)
    
    if model_name in ['inception_v3','tf2torch_adv_inception_v3','tf2torch_ens4_adv_inc_v3','tf2torch_ens3_adv_inc_v3','tf2torch_ens_adv_inc_res_v2']:
        img_size = 299
    else:
        img_size = 224
    transforms = T.Compose([T.Resize(img_size), T.ToTensor()])
    X = ImageNet(adv_dir, input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    
    for i in ["HGD","Bit_red","FD","JPEG","NRP","RP","RS"]:
        
        sum = 0
        for images, _, gt_cpu in data_loader:
            gt = gt_cpu.cuda()
            images = images.cuda()        
        
            if i == "HGD":
            #HGD  
                model=get_HGD_model().cuda().eval()
                image = images
            elif i == "Bit_red":
                image = bit_depth_red(images,3) #levels=[7,6,5,4,3,2] #bit depths 
            elif i == "FD":
                image = FD_jpeg_encode(images)#FD
            elif i == "JPEG":
                image = JPEG_compression(images,70) #levels=[90,80,70,60,50,40,30]#JPEG compression ratios             

            elif i == "NRP":
                netG=nrp_model().cuda().eval()
                image = netG(images).detach()
            elif i == "RP":
                image = randomize_input(images) 
            elif i == "RS":
                defense = smooth_defense(skip=100, N=100, batch_size=400, path='./models/defense/rs_imagenet/resnet50/noise_0.25/checkpoint.pth.tar', dataset='imagenet', sigma=0.25, alpha=0.001)
            else:
                print("no wrong")

            with torch.no_grad():

                if i == "RS":
                    sum += defense(images,gt)
                else:

                    sum += (model(image).argmax(1) != (gt)).detach().sum().cpu()# 其他防御.

        print(i +"  " +model_name + '  acu = {:.2%}'.format(sum / 1000.0))
        with open(file_path, 'a') as f:
            f.write(i +" " +model_name + '  acu = {:.2%}\n'.format(sum / 1000.0))
        print("===================================================")
        with open(file_path, 'a') as f:
            f.write("===================================================\n")
    
    
    
    
def verify_ensmodels(model_name, path):
    model = get_model(model_name, path)
    
    if model_name in ['inception_v3','tf2torch_adv_inception_v3','tf2torch_ens4_adv_inc_v3','tf2torch_ens3_adv_inc_v3','tf2torch_ens_adv_inc_res_v2']:
        img_size = 299
    else:
        img_size = 224
    transforms = T.Compose([T.Resize(img_size), T.ToTensor()])
    X = ImageNet(adv_dir, input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum = 0
    for images, _, gt_cpu in data_loader:
        gt = gt_cpu.cuda()
        images = images.cuda()
        with torch.no_grad():
            # print(sum)
            sum += (model(images)[0].argmax(1) != (gt+1)).detach().sum().cpu()

    print(model_name + '  acu = {:.2%}'.format(sum / 1000.0))
    with open(file_path, 'a') as f:
        f.write(model_name + '  acu = {:.2%}\n'.format(sum / 1000.0))

def main():
    model_names = ['resnet50'] #打算攻击的模型
    #model_names = ['inception_v3', 'inception_v4']
    model_names_ens = ['tf2torch_ens4_adv_inc_v3','tf2torch_ens3_adv_inc_v3','tf2torch_ens_adv_inc_res_v2'] # You can download the pretrained ens_models from https://github.com/ylhz/tf_to_pytorch_model
    models_path = './models/'
    for model_name in model_names:

        verify(model_name, models_path)
        print("===================================================")
        with open(file_path, 'a') as f:
            f.write("===================================================\n")
    for model_name in model_names_ens: # When we validate the ens model, we should change gt to gt+1 as the ground truth label.
       verify_ensmodels(model_name, models_path)
       print("===================================================")
       with open(file_path, 'a') as f:
           f.write("===================================================\n")
if __name__ == '__main__':
    print("  ")
    print("  ")
    print("  ")
    print(adv_dir)
    with open(file_path, 'w') as f:
        f.write(adv_dir+'\n')
    main()

#print("use %s attack %s generate adversarial to attack %s"%(opt.attack,opt.name,opt.black_name))