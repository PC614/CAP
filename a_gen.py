
#  一个攻击算法  攻击4个代理模型 IC3 resnet50 den121 vit 跟一个 集成模型 (resnet50 den121 vit)

import os

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--attack', default='ni_fgsm', type=str, help='fgsm, ifgsm, mifgsm, vmi_fgsm, ssi,ssmi, si_fgsm,smi_fgsm,admix,mmi_fgsm')

# 攻击的模型 densenet121、inception_v3、resnet50、swin-t、vit -- "Ensemble_Model"---- tf2torch_ens3_adv_inc_v3 、tf2torch_ens4_adv_inc_v3 、tf2torch_ens_adv_inc_res_v2

opt = parser.parse_args()


os.system('python ge_adv.py --attack cap --name densenet121')
os.system('python ge_adv.py --attack cap --name inception_v3')

os.system('python ge_adv.py --attack cap --name resnet50')
os.system('python ge_adv.py --attack cap --name vit')

os.system('python ge_adv.py --attack cap --name Ensemble_Model')
#os.system('python ge_adv.py --attack pgn --name Ensemble_Model')