import os

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--attack', default='pgn', type=str, help='fgsm, ifgsm, mifgsm, vmi_fgsm, ssi,ssmi, si_fgsm,smi_fgsm,admix,mmi_fgsm')

# 攻击的模型 densenet121、inception_v3、resnet50、swin-t、vit -- "Ensemble_Model"---- tf2torch_ens3_adv_inc_v3 、tf2torch_ens4_adv_inc_v3 、tf2torch_ens_adv_inc_res_v2

parser.add_argument('--name', default='inception_v3', type=str, help='model')
opt = parser.parse_args()





"""
检测一个攻击   即攻击算法固定
"""

        
    
result = './output/%s_result.txt'%(opt.attack)
    
os.system('python verify.py --attack %s --name inception_v3| tee -a %s'%(opt.attack,result))  #表示用 --attack攻击 --nane生成的对抗样本 去测试
#os.system('python verify.py --attack %s --name densenet121| tee -a %s'%(opt.attack,result))
#os.system('python verify.py --attack %s --name resnet50| tee -a %s'%(opt.attack,result))
#os.system('python verify.py --attack %s --name vit| tee -a %s'%(opt.attack,result))



#os.system('python verify.py --attack %s --name Ensemble_Model| tee -a %s'%(opt.attack,result))