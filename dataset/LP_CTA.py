import pickle
import numpy as np
from PIL import Image
import os

import torch
from torchvision import transforms, utils
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import resize

from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.transforms.abstract_transforms import Compose, RndTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform, ResizeTransform
from batchgenerators.transforms.color_transforms import BrightnessTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor

join = os.path.join


class LP_CTA_Dataset(Dataset):
    def __init__(self, keys, args, mode='train'):
        super().__init__()
        self.patch_size = (args.img_size, args.img_size)
        print(f'patch size: {self.patch_size}')
        self.files = []
        self.mode = mode

        for f in os.listdir(args.data_dir):
            if f.split("_frame")[0] in keys:
                slices = subfiles(join(args.data_dir, f))
                for sl in slices:
                    self.files.append(sl)  # Dataset folder

        print(f'dataset length: {len(self.files)}') 

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        label = Image.open(self.files[index].replace('imgs', 'annotations'))   
        label = np.asarray(label)/255
        # scribble = Image.open(self.files[index].replace('imgs/', 'scribbles/'))
        # scribble = np.asarray(scribble)

        img = np.asarray(img).astype(np.float32).transpose([2, 0, 1])
        # img = np.asarray(img).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        # print(img.shape, label.shape)
        # 绘制label
        # import matplotlib.pyplot as plt
        # plt.imshow(label)  
        # plt.show()


        if self.mode == 'contrast':
            img1, img2 = self.transform_contrast(img)
            return img1, img2
        else:
            img, label = self.transform(img, label)
            return img, label

    def transform_contrast(self, img):
        # the image and label should be [batch, c, x, y, z], this is the adapatation for using batchgenerators :)
        data_dict = {'data': img[None]}
        tr_transforms = [  # CenterCropTransform(crop_size=target_size),
            BrightnessTransform(mu=1, sigma=1, p_per_sample=0.5),
            GammaTransform(p_per_sample=0.5),
            GaussianNoiseTransform(p_per_sample=0.5),
            ResizeTransform(target_size=self.patch_size, order=1),  # resize
            MirrorTransform(axes=(1,)),
            SpatialTransform(patch_size=self.patch_size, random_crop=False,
                             patch_center_dist_from_border=self.patch_size[0] // 2,
                             do_elastic_deform=True, alpha=(100., 350.), sigma=(40., 60.),
                             do_rotation=True, p_rot_per_sample=0.5,
                             angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                             scale=(0.5, 1.9), p_scale_per_sample=0.5,
                             border_mode_data="nearest", border_mode_seg="nearest"),
        ]

        train_transform = Compose(tr_transforms)
        data_dict = train_transform(**data_dict)
        img1 = data_dict.get('data')[0]
        data_dict = train_transform(**data_dict)
        img2 = data_dict.get('data')[0]
        return img1, img2

    def transform(self, img, label):
        # normalize to [0, 1]
        data_dict = {'data': img[None], 'seg': label[None, None]}
        if self.mode == 'train':
            aug_list = [  # CenterCropTransform(crop_size=target_size),
                BrightnessTransform(mu=1, sigma=1, p_per_sample=0.5),
                GammaTransform(p_per_sample=0.5),
                GaussianNoiseTransform(p_per_sample=0.5),
                ResizeTransform(target_size=self.patch_size, order=1),  # resize
                MirrorTransform(axes=(1,)),
                SpatialTransform(patch_size=self.patch_size, random_crop=False,
                                 patch_center_dist_from_border=self.patch_size[0] // 2,
                                 do_elastic_deform=True, alpha=(100., 350.), sigma=(40., 60.),
                                 do_rotation=True, p_rot_per_sample=0.5,
                                 angle_x=(-0.1, 0.1), angle_y=(0, 1e-8), angle_z=(0, 1e-8),
                                 scale=(0.5, 1.9), p_scale_per_sample=0.5,
                                 border_mode_data="nearest", border_mode_seg="nearest"),
                NumpyToTensor(),
            ]

            aug = Compose(aug_list)
        else:
            aug_list = [
                ResizeTransform(target_size=self.patch_size, order=1),
                NumpyToTensor(),
            ]
            aug = Compose(aug_list)
        # print('data_shape',data_dict.get('data').shape)
        # print('seg_shape',data_dict.get('seg').shape)
        # print(aug)
        data_dict = aug(**data_dict)
        img = data_dict.get('data')[0]
        label = data_dict.get('seg')[0]
        return img, label



if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/LP_CTA/imgs/')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--src_dir', type=str, default='dataset/LP_CTA/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument("--tr_size", type=int, default=1)
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
    args = parser.parse_args()
    split_dir = os.path.join(args.src_dir, "splits.pkl")
    with open(split_dir, "rb") as f:
        splits = pickle.load(f)
    tr_keys = splits[args.fold]['train']
    val_keys = splits[args.fold]['val']
    test_keys = splits[args.fold]['test']

    if args.tr_size < len(tr_keys):
        tr_keys = tr_keys[0:args.tr_size]

    print(tr_keys)
    print(val_keys)
    print(test_keys)

    dataset = LP_CTA_Dataset(keys=tr_keys, mode='train', args=args)
    print(len(dataset))
    img1, img2 = dataset[0]
    print(np.unique(img1))
    print(np.unique(img2)) # [0. 1.]
    print(img1.shape, img2.shape)   # torch.Size([3, 224, 224]) torch.Size([1, 224, 224])
    plt.imshow(img1[0], cmap='gray')
    plt.show()
    plt.imshow(img2[0],cmap='gray')
    plt.show()

    # test_sampler = None
    # for key in test_keys:
    #     args.img_size = 224
    #     test_ds = LP_CTA_Dataset(keys=key, mode='val', args=args)
    #     data_loader = torch.utils.data.DataLoader(
    #         test_ds, batch_size=args.batch_size, shuffle=False,
    #         num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=False
    #     )
    #     print(key)
    #     for i, tup in enumerate(data_loader):
    #         if args.gpu is not None:
    #             img = tup[0].float().cuda(args.gpu, non_blocking=True)
    #             label = tup[1].long().cuda(args.gpu, non_blocking=True)
    #             print(img.shape, label.shape)