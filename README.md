# AC2D
This repository is the official PyTorch implementation including training and testing of IJCV paper ["Facial Action Unit Detection by Adaptively Constraining Self-Attention and Causally Deconfounding Sample"](https://arxiv.org/pdf/2410.01251.pdf)

# Getting Started
## Installation
- Clone this repo:
```
git clone https://github.com/ZhiwenShao/AC2D
cd AC2D
```
- Install requirements:
```
conda create -n ac2d python=3.11.3
conda activate ac2d
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Datasets
[BP4D](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html), [DISFA](http://mohammadmahoor.com/disfa), [GFT](https://osf.io/7wcyz), [BP4D+](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html), [Aff-Wild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2)

Follow the guidelines in [PyTorch-JAANet](https://github.com/ZhiwenShao/PyTorch-JAANet) for data preparation and preprocessing

## Training
- Train on BP4D with the first two folds for training and the third fold for testing:
```
python train.py --net='restv2_tiny_ac2d' --run_name='BP4D_combine_1_2' --train_path_prefix='data/list/BP4D_combine_1_2' --test_path_prefix='data/list/BP4D_part3' --flip_reflect='data/list/reflect_49.txt' --au_num=12 --dataset_name='BP4D' --train_batch_size=24 --eval_batch_size=20 --gpu_id=0
```
- Train on DISFA with the first two folds for training and the third fold for testing, using the the well-trained BP4D model for initialization:
```
python train_disfa.py --net='restv2_tiny_ac2d' --run_name='DISFA_combine_1_2' --train_path_prefix='data/list/DISFA_combine_1_2' --test_path_prefix='data/list/DISFA_part3' --flip_reflect='data/list/reflect_49.txt' --au_num=8 --dataset_name='DISFA' --train_batch_size=24 --eval_batch_size=20 --gpu_id=0 --pretrain_net_path='restv2_tiny_ac2d/BP4D_combine_1_3' --pretrain_epoch=11
```
- Train on GFT:
```
python train.py --net='restv2_tiny_ac2d' --run_name='GFT' --train_path_prefix='data/list/GFT_train' --test_path_prefix='data/list/GFT_test' --flip_reflect='data/list/reflect_49.txt' --au_num=10 --dataset_name='GFT' --train_batch_size=24 --eval_batch_size=20 --gpu_id=0
```
- Train on the whole BP4D and test on the whole BP4D+:
```
python train.py --net='restv2_tiny_ac2d' --run_name='BP4D_cross_BP4D_plus' --train_path_prefix='data/list/BP4D' --test_path_prefix='data/list/BP4D_plus' --flip_reflect='data/list/reflect_49.txt' --au_num=12 --dataset_name='BP4D' --train_batch_size=24 --eval_batch_size=20 --gpu_id=0
```
- Train on Aff-Wild2:
```
python train.py --net='restv2_tiny_ac2d' --run_name='Aff-Wild2' --train_path_prefix='data/list/Aff-Wild2_train' --test_path_prefix='data/list/Aff-Wild2_validation' --flip_reflect='data/list/reflect_49.txt' --au_num=10 --dataset_name='Aff-Wild2' --train_batch_size=24 --eval_batch_size=20 --gpu_id=0
```

## Testing
- Test the models saved in different epochs:
```
python test.py --net='restv2_tiny_ac2d' --run_name='BP4D_combine_1_2' --test_path_prefix='data/list/BP4D_part3' --flip_reflect='data/list/reflect_49.txt' --au_num=12 --dataset_name='BP4D' --eval_batch_size=20 --gpu_id=0 --start_epoch=1 --epochs=20  
```
- Visualize self-attention maps:
```
python test.py --net='restv2_tiny_ac2d' --run_name='BP4D_combine_1_2' --test_path_prefix='data/list/BP4D_part3_sample' --flip_reflect='data/list/reflect_49.txt' --au_num=12 --dataset_name='BP4D' --eval_batch_size=20 --gpu_id=0 --start_epoch=1 --epochs=20 --pred_AU=False --vis_attention=True
```
- Compute FLOPs and \#Params.:
```
python test.py --net='restv2_tiny_ac2d' --run_name='BP4D_combine_1_2' --test_path_prefix='data/list/BP4D_part3_sample' --flip_reflect='data/list/reflect_49.txt' --au_num=12 --dataset_name='BP4D' --eval_batch_size=1 --gpu_id=0 --start_epoch=20 --epochs=20 --pred_AU=False --cal_flops=True
```

## Citation
If you use this code for your research, please cite our paper:
```
@article{shao2024facial,
  title={Facial Action Unit Detection by Adaptively Constraining Self-Attention and Causally Deconfounding Sample},
  author={Shao, Zhiwen and Zhu, Hancheng and Zhou, Yong and Xiang, Xiang and Liu, Bing and Yao, Rui and Ma, Lizhuang},
  journal={International Journal of Computer Vision},
  year={2024},
  publisher={Springer}
}
```
Should you have any questions, just contact with us through email zhiwen_shao@cumt.edu.cn
