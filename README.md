# CAP
This repository contains the PyTorch code for the manuscript:

Enhancing Adversarial Transferability via Curvature-Aware Penalization

## Requirements
* python == 3.8.20
* pytorch == 2.4.0
* torchvision == 0.19.0
* numpy == 1.24.4
* pandas == 2.0.3
* opencv == 4.10.0
* scipy == 1.10.1
* pillow == 10.4.0
* pretrainedmodels == 0.7.4
* tqdm == 4.66.5
* imageio == 2.33.1


## Qucik Start
### Prepare the data and models.
1. You can download the ImageNet-compatible dataset from [here](https://github.com/Zhijin-Ge/STM/tree/main/dataset) and put the data in **'./dataset/'**.

2. The normally trained models (i.e., Inc-v3, Inc-v4, IncRes-v2, Res-50, Res-101, Res-100) are from "pretrainedmodels", if you use it for the first time, it will download the weight of the model automatically, just wait for it to finish. 

3. You can download the adversarially trained models (i.e, ens3_adv_inc_v3, ens4_adv_inc_v3, ens_adv_inc_res_v2) from [here](https://drive.google.com/drive/folders/1O_3HIHFeAjycR0z_px7Mb_XqKqyDm7eK?usp=drive_link) and put the model in **'./model/'**.

### CAP Attack Method
The traditional baseline attacks and our proposed CAP attack methods are in the file __"attack_method.py"__.


### Runing attack
1. You can run our proposed attack as follows. 
```
python ge_adv.py --attack cap --name inception_v3
```
We also provide the implementations of other baseline attack methods in our code, just change cap to the corresponding attack methods in the code.

2. The generated adversarial examples would be stored in the directory **./adv_xx_xx**. Then run the file **verify.py** to evaluate the success rate of each model used in the paper:
```
python verify.py --attack cap --name inception_v3
```





