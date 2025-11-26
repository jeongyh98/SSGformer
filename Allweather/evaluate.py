import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import time
import util
import yaml
import argparse

from tqdm import tqdm
from glob import glob
from natsort import natsorted
from skimage import img_as_ubyte

import sys
sys.path.append(os.getcwd())
import basicsr.models.archs.SSGformer_arch as M
from compute_psnr import compute_metrics

def create_dir(args, res_path, gt_path):

    if 'drop' in gt_path: base_name = 'drop'
    elif 'Snow' in  gt_path: base_name = 'snow'
    else: base_name = 'rain'
    
    ckpt_name = args.checkpoint.split('.')[0]
    result_dir = os.path.join(res_path, args.model, ckpt_name, base_name)
    if os.path.exists(result_dir):
        index = 1
        while os.path.exists(f"{res_path}/{args.model}/{ckpt_name}/{base_name}_ver_{index}"):
            index += 1
        result_dir = f"{res_path}/{args.model}/{ckpt_name}/{base_name}_ver_{index}"
    os.makedirs(result_dir, exist_ok=True)
    return result_dir, ckpt_name

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image Deraining using Restormer')

    parser.add_argument('--data', type=str, help='image directory')
    parser.add_argument('--model', type=str, help='experiment name')
    parser.add_argument('--checkpoint', type=str, help='checkpoint path')

    args = parser.parse_args()

    ############################################################
    # configuration
    yaml_path = './Allweather/Options'
    exp_path = './experiments'
    res_path = './evaluation'
    ############################################################
    
    # load yaml file
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    yaml_file = os.path.join(yaml_path, f'{args.model}.yml')
    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

    # load model
    s = x['network_g'].pop('type')
    model = getattr(M, s)
    model_restoration = model(**x['network_g'])

    # load checkpoint
    ckpt_path = os.path.join(exp_path, args.model, 'models', args.checkpoint)
    ckpt_dict = torch.load(ckpt_path)
    model_restoration.load_state_dict(ckpt_dict['params'])
    print("===>Testing using checkpoint: ", ckpt_path)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    # load data
    inp_path = os.path.join(args.data, 'input')
    gt_path = os.path.join(args.data, 'gt')
    files = natsorted(glob(os.path.join(inp_path, '*.png')) + glob(os.path.join(inp_path, '*.jpg')))

    # create directory
    result_dir, ckpt_name = create_dir(args, res_path, gt_path)

    # evaluation & visualization
    factor = 8
    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
  
            img = np.float32(util.load_img(file_))
            img = torch.from_numpy(img).permute(2,0,1)
            input_ = img.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
        
            # import pdb; pdb.set_trace()
            mask_lq = np.load(file_.replace('input','mask').replace('jpg','npy').replace('png','npy'))
            mask_lq = np.expand_dims(mask_lq, axis=2)
            mask_lq = torch.from_numpy(mask_lq).permute(2,0,1)
            mask_lq = mask_lq.unsqueeze(0).cuda()
            mask_lq = F.pad(mask_lq, (0,padw,0,padh), 'reflect')

            # results
            try:
                restored = model_restoration(input_, mask_lq, file_)
            except:
                restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:,:,:h,:w]
            restored = torch.clamp(restored/255.,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            
            util.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(restored))

    if 'snow' in result_dir:
        snow_mean_mse, snow_mean_psnr, snow_mean_ssim = compute_metrics(result_dir, gt_path)
        rain_mean_mse, rain_mean_psnr, rain_mean_ssim = compute_metrics(result_dir.replace('snow', 'rain'), gt_path.replace('Snow100K-L', 'test1'))
        drop_mean_mse, drop_mean_psnr, drop_mean_ssim = compute_metrics(result_dir.replace('snow', 'drop'), gt_path.replace('Snow100K-L', 'rain_drop_test'))

        mean_psnr = np.mean([snow_mean_psnr, rain_mean_psnr, drop_mean_psnr])
        mean_ssim = np.mean([snow_mean_ssim, rain_mean_ssim, drop_mean_ssim])

        # log file
        log_file = f"{res_path}/{args.model}/{ckpt_name}/eval.txt"
        f = open(log_file, "a+")
        local_time = time.strftime('%Y.%m.%d - %H:%M:%S')
                
        f.write('%s\n' %('-'*88))
        f.write(f'{"model name":^18}:   {f"{args.model}":<30}\n')
        f.write(f'{"checkpoint":^18}:   {f"{ckpt_name}":<30}\n')
        f.write(f'{"local time":^18}:   {f"{local_time}":<30}\n')
        f.write('%s\n' %('-'*88))
        
        f.write(f'{"metric":>10} |{"snow":^17}|{"rain":^17}|{"drop":^17}|{"avg":^17}\n')
        f.write('%s\n' %('-'*88))
        f.write(f'{"psnr":>12} |{snow_mean_psnr:^17.4f}|{rain_mean_psnr:^17.4f}|{drop_mean_psnr:^17.4f}|{mean_psnr:^15.4f}\n')
        f.write(f'{"ssim":>12} |{snow_mean_ssim:^17.4f}|{rain_mean_ssim:^17.4f}|{drop_mean_ssim:^17.4f}|{mean_ssim:^15.4f}\n\n')
        f.close()

        