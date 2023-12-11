# AutoSAM 
This repo is pytorch implementation of paper "How to Efficiently Adapt Large Segmentation Model(SAM) to Medical Image Domains" by Xinrong Hu et al.

[[`Paper`](https://arxiv.org/pdf/2306.13731.pdf)]

![](./autosam.png)
## Setup
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. 

clone the repository locally:

```
git clone git@github.com:xhu248/AutoSAM.git
```
and install requirements:
```
cd AutoSAM; pip install -e .
```
Download the checkpoints from [SAM](https://github.com/facebookresearch/segment-anything#model-checkpoints) and place them under AutoSAM/

## Dataset

The original ACDC data files can be dowonloaded at [Automated Cardiac Diagnosis Challenge ](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).
The data is provided in nii.gz format. We convert them into PNG files as SAM requires RGB input. 
The processed data can be downloaded [here](https://drive.google.com/drive/folders/1RcpWYJ7EkwPiCR9u6HRrg7JHQ_Dr7494?usp=drive_link)

## How to use
### Finetune CNN decoder
```
python scripts/main_feat_seg.py --src_dir ${ACDC_folder} \
--data_dir ${ACDC_folder}/imgs/ --save_dir ./${output_dir}  \
--b 4 --dataset ACDC --gpu ${gpu} \
--fold ${fold} --tr_size ${tr_size}  --model_type ${model_type} --num_classes 4
```
${tr_size} decides how many volumes used in the training; ${model_type} is selected from vit_b (default), vit_l, and vit_h;

### Finetune AutoSAM
```
python scripts/main_autosam_seg.py --src_dir ${ACDC_folder} \
--data_dir ${ACDC_folder}/imgs/ --save_dir ./${output_dir}  \
--b 4 --dataset ACDC --gpu ${gpu} \
--fold ${fold} --tr_size ${tr_size}  --model_type ${model_type} --num_classes 4
```
**eg. my terminal**
#### ACDC
```windows
python scripts/main_autosam_seg.py --src_dir dataset\ACDC --data_dir dataset\ACDC\imgs\ --save_dir .\output_dir\ACDC --b 4 --dataset ACDC --gpu 0 --fold 1 --tr_size 1 --model_type vit_l --num_classes 4
```
```ubuntu
python scripts/main_autosam_seg.py --src_dir dataset/ACDC --data_dir dataset/ACDC/imgs/ --save_dir ./output_dir/ACDC --b 4 --dataset ACDC --gpu 1 --fold 1 --tr_size 1 --model_type vit_l --num_classes 4

```
#### LP_CTA
python scripts/main_autosam_seg.py --src_dir dataset/LP_CTA --data_dir dataset/LP_CTA/imgs/ --save_dir ./output_dir/LP_CTA --b 4 --dataset LP_CTA --gpu 1 --fold 1 --tr_size 1 --model_type vit_l --num_classes 2

This repo also supports distributed training
```
python scripts/main_autosam_seg.py --src_dir ${ACDC_folder} --dist-url 'tcp://localhost:10002' \
--data_dir ${ACDC_folder}/imgs/ --save_dir ./${output_dir} \
--multiprocessing-distributed --world-size 1 --rank 0  -b 4 --dataset ACDC \
--fold ${fold} --tr_size ${tr_size}  --model_type ${model_type} --num_classes 4
```
## Notes( by Mancy)
数据集缺失的条目：
- 'dataset/LP_CTA/annotations/74/patient_074_frame_124.png'
- 'dataset/LP_CTA/annotations/169/patient_169_frame_112.png'


## Citation
If you find our codes useful, please cite
```
@article{hu2023efficiently,
  title={How to Efficiently Adapt Large Segmentation Model (SAM) to Medical Images},
  author={Hu, Xinrong and Xu, Xiaowei and Shi, Yiyu},
  journal={arXiv preprint arXiv:2306.13731},
  year={2023}
}
```
