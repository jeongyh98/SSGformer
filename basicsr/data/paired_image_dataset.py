from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import paired_random_crop_DP, random_augmentation
from basicsr.utils import img2tensor, padding_DP

import random
import numpy as np
import torch
import math
import os
from PIL import Image


class AllweatherPairedImage(data.Dataset):
    def __init__(self, opt, phase=None):
        super(AllweatherPairedImage, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.data_root, self.data_root = opt['dataroot_gt'], opt['dataroot_lq']
        
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

        self._init_ids()
        self._merge_ids()

    def _init_ids(self):
        self._init_snow_ids()
        self._init_rain_ids()
        self._init_drop_ids()

    def _init_rain_ids(self):
        temp_ids = []
        rain_dir = os.path.join(self.data_root, 'HeavyRain', 'input')
        rain_list = os.listdir(rain_dir)
        rain_sample = len(rain_list) if 9000> len(rain_list) else 9000
        rain_list = random.sample(rain_list, rain_sample)
        rain_list = rain_list * (1 + math.ceil(self.snow_sample / rain_sample))
        rain_list = rain_list[0: self.snow_sample]
        temp_ids += [os.path.join(rain_dir, id_) for id_ in rain_list]
        self.rain_ids = [{"clean_id": x, "we_type": 'rain'} for x in temp_ids]
        self.num_rain = len(self.rain_ids)
        print("[TRAIN] Total HeavyRain Ids : {}".format(self.num_rain))

    def _init_drop_ids(self):
        temp_ids = []
        drop_dir = os.path.join(self.data_root, 'RainDrop', 'input')
        drop_list = os.listdir(drop_dir)
        drop_sample = len(drop_list) if 9000 > len(drop_list) else 9000
        drop_list = random.sample(drop_list, drop_sample)
        drop_list = drop_list * (1 + math.ceil(self.snow_sample / drop_sample))
        drop_list = drop_list[0: self.snow_sample]
        temp_ids += [os.path.join(drop_dir, id_) for id_ in drop_list]
        self.drop_ids = [{"clean_id": x, "we_type": 'drop'} for x in temp_ids]
        self.num_drop = len(self.drop_ids)
        print("[TRAIN] Total RainDrop Ids : {}".format(self.num_drop))

    def _init_snow_ids(self):
        temp_ids = []
        snow_dir = os.path.join(self.data_root, 'Snow', 'input')
        snow_list = os.listdir(snow_dir)
        self.snow_sample = len(snow_list) if 9000 > len(snow_list) else 9000
        snow_list = random.sample(snow_list, self.snow_sample)
        temp_ids += [os.path.join(snow_dir, id_) for id_ in snow_list]
        self.snow_ids = [{"clean_id": x, "we_type": 'snow'} for x in temp_ids]
        self.num_snow = len(self.snow_ids)
        print("[TRAIN] Total Snow Ids : {}".format(self.num_snow))

    def _get_gt_name(self, input_name):
        gt_name = input_name.replace('input', 'gt')
        return gt_name
    
    def _merge_ids(self):
        self.sample_ids = []
        self.sample_ids += self.rain_ids
        self.sample_ids += self.drop_ids
        self.sample_ids += self.snow_ids
        print(len(self.sample_ids))

    def __getitem__(self, index):

        scale = self.opt['scale']
        
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        sample = self.sample_ids[index]
        de_id = sample["we_type"]
        img_lq = np.array(Image.open(sample["clean_id"]).convert('RGB'))[...,[2,1,0]]
        mask_lq = np.load(sample["clean_id"].replace('input','mask').replace('jpg','npy').replace('png','npy'))
        mask_lq = np.expand_dims(mask_lq, axis=2)
        clean_name = self._get_gt_name(sample["clean_id"])
        img_gt = np.array(Image.open(clean_name).convert('RGB'))[...,[2,1,0]]
        _, h, w =img_gt.shape
        
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, mask_lq, img_lq = padding_DP(img_gt, mask_lq, img_lq, gt_size)
            # random crop
            img_gt, mask_lq, img_lq = paired_random_crop_DP(img_gt, mask_lq, img_lq, gt_size, scale, clean_name)
            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, mask_lq, img_lq = random_augmentation(img_gt, mask_lq, img_lq)
            
            if len(mask_lq.shape) == 2:
                mask_lq = np.expand_dims(mask_lq, axis=2)
        elif "val" in self.opt['phase']:            
            factor = 16
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            img_gt = F.pad(img_gt, (0,padw,0,padh), 'reflect')
            mask_lq = F.pad(mask_lq, (0,padw,0,padh), 'reflect')
            img_lq = F.pad(img_lq, (0,padw,0,padh), 'reflect')
       
       # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        mask_lq = torch.from_numpy(mask_lq).permute(2,0,1)
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        return {
            'lq': img_lq,
            'mask': mask_lq,
            'gt': img_gt,
            'lq_path': sample["clean_id"],
            'gt_path': clean_name,
            'label': de_id
        }

    def __len__(self):
        return len(self.sample_ids)
    
class AllWeatherTestPairedImage(data.Dataset):
    def __init__(self, opt, phase=None):
        super(AllWeatherTestPairedImage, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.data_root, self.data_root = opt['dataroot_gt'], opt['dataroot_lq']
        
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

        if phase is not None:
            self.we_type = phase
        
        self._init_ids()
        self._merge_ids()

    def _init_ids(self):
        if 'snow' in self.we_type:
            self._init_snow_ids()
        if 'test' in self.we_type:
            self._init_rain_ids()
        if 'drop' in self.we_type:
            self._init_drop_ids()

    def _init_rain_ids(self):
        temp_ids = []
        rain_dir = os.path.join(self.data_root)
        rain_list = sorted(os.listdir(rain_dir))
        temp_ids += [os.path.join(rain_dir, id_) for id_ in rain_list]
        self.rain_ids = [{"clean_id": x, "we_type": 'rain'} for x in temp_ids]
        self.rain_counter = 0
        self.num_rain = len(self.rain_ids)
        print("[TEST] Total HeavyRain Ids : {}".format(self.num_rain))

    def _init_drop_ids(self):
        temp_ids = []
        drop_dir = os.path.join(self.data_root)
        drop_list = sorted(os.listdir(drop_dir))
        temp_ids += [os.path.join(drop_dir, id_) for id_ in drop_list]
        self.drop_ids = [{"clean_id": x, "we_type": 'drop'} for x in temp_ids]
        self.rain_counter = 0
        self.num_drop = len(self.drop_ids)
        print("[TEST] Total RainDrop Ids : {}".format(self.num_drop))
    
    def _init_snow_ids(self):
        temp_ids = []
        snow_dir = os.path.join(self.data_root)
        snow_list = sorted(os.listdir(snow_dir))
        temp_ids += [os.path.join(snow_dir, id_) for id_ in snow_list]
        self.snow_ids = [{"clean_id": x, "we_type": 'snow'} for x in temp_ids]
        self.snow_counter = 0
        self.num_snow = len(self.snow_ids)
        print("[TEST] Total Snow Ids : {}".format(self.num_snow))

    def _get_gt_name(self, input_name):
        gt_name = input_name.replace("input", "gt").replace('_rain.png', '_clean.png')
        return gt_name
    
    def _merge_ids(self):
        self.sample_ids = []
        if "test" in self.we_type:
            self.sample_ids += self.rain_ids
        if "drop" in self.we_type:
            self.sample_ids += self.drop_ids
        if "snow" in self.we_type:
            self.sample_ids += self.snow_ids
        print(len(self.sample_ids))
        
    def __getitem__(self, index):

        scale = self.opt['scale']
        
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        sample = self.sample_ids[index]
        de_id = sample["we_type"]
        img_lq = np.array(Image.open(sample["clean_id"]).convert('RGB'))[...,[2,1,0]]
        mask_lq = np.load(sample["clean_id"].replace('input','mask').replace('jpg','npy').replace('png','npy'))
        mask_lq = np.expand_dims(mask_lq, axis=2)
        clean_name = self._get_gt_name(sample["clean_id"])
        img_gt = np.array(Image.open(clean_name).convert('RGB'))[...,[2,1,0]]
        
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, mask_lq, img_lq = padding_DP(img_gt, mask_lq, img_lq, gt_size)
            # random crop
            img_gt, mask_lq, img_lq = paired_random_crop_DP(img_gt, mask_lq, img_lq, gt_size, scale, clean_name)
            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, mask_lq, img_lq = random_augmentation(img_gt, mask_lq, img_lq)
            
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        
        mask_lq = torch.from_numpy(mask_lq).permute(2,0,1)
        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        return {
            'lq': img_lq,
            'mask': mask_lq,
            'gt': img_gt,
            'lq_path': sample["clean_id"],
            'gt_path': clean_name,
            'label': de_id,
        }

    def __len__(self):
        return len(self.sample_ids)