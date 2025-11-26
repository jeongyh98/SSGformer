import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import torch.nn.functional as F
from functools import partial

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network
        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()
        self.psnr_best = -1
        
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            raise ValueError('pixel loss are None.')
        if train_opt.get('seq_opt'):
#            from audtorch.metrics.functional import pearsonr
#            self.cri_seq = pearsonr
            self.cri_seq = self.pearson_correlation_loss #
        self.cri_celoss = torch.nn.CrossEntropyLoss()
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        
    def pearson_correlation_loss(self, x1, x2):
        assert x1.shape == x2.shape
        b, c = x1.shape[:2]
        dim = -1
        x1, x2 = x1.reshape(b, -1), x2.reshape(b, -1)
        x1_mean, x2_mean = x1.mean(dim=dim, keepdims=True), x2.mean(dim=dim, keepdims=True)
        numerator = ((x1 - x1_mean) * (x2 - x2_mean)).sum( dim=dim, keepdims=True )
        
        std1 = (x1 - x1_mean).pow(2).sum(dim=dim, keepdims=True).sqrt() 
        std2 = (x2 - x2_mean).pow(2).sum(dim=dim, keepdims=True).sqrt()
        denominator = std1 * std2
        corr = numerator.div(denominator + 1e-6)
        return corr

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'label' in data:
            self.label = data['label']
        if 'mask' in data:
            self.mask = data['mask'].to(self.device)
        if self.mixing_flag: # false
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'mask' in data:
            self.mask = data['mask'].to(self.device)

    def check_inf_nan(self, x):
        x[x.isnan()] = 0
        x[x.isinf()] = 1e7
        return x
    
    def compute_correlation_loss(self, x1, x2):
        b, c = x1.shape[0:2]
        x1 = x1.view(b, -1)
        x2 = x2.view(b, -1)
#        print(x1, x2)
        pearson = (1. - self.cri_seq(x1, x2)) / 2.
        return pearson[~pearson.isnan()*~pearson.isinf()].mean()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.mask, )

        loss_dict = OrderedDict()

        l1_loss = self.cri_pix(self.output, self.gt)
        loss_dict['l1_loss'] = l1_loss
        '''
        l_mask = self.cri_pix(self.pred_mask, self.gt - self.output.detach())
        loss_dict['l_mask'] = l_mask
        '''
        
        l_pear = self.compute_correlation_loss(self.output, self.gt)
        loss_dict['l_pred'] = l_pear
        loss_total = l1_loss + l_pear #+ 0.01*l_pred#+ l_mask
        loss_total.backward()
        
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01, error_if_nonfinite=False)
        self.optimizer_g.step()

        self.log_dict, self.loss_total = self.reduce_loss_dict(loss_dict)
        self.loss_dict = loss_dict
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):       
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        mask = F.pad(self.mask, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # original_size = self.original_size
        self.nonpad_test(img, mask)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None, mask=None): #, original_size=None):
        if img is None:
            img = self.lq
        if img is None:
            mask = self.mask
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)#[:,:, :original_size[0], :original_size[1]]
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img, mask)#[:,:, :original_size[0], :original_size[1]]
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx >= 60:
                break
            self.feed_data(val_data)
            test()
            visuals = self.get_current_visuals()

            # tentative for out of GPU memory
            del self.lq
            del self.mask
            del self.output
            torch.cuda.empty_cache()

            if with_metrics:
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = max(current_metric, self.metric_results[metric])

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if metric == 'psnr' and value >= self.psnr_best:
                self.save(0, current_iter, best=True)
                self.psnr_best = value
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            if 'Snow' in dataset_name:
                for metric, value in self.metric_results.items():
                    tb_logger.add_scalar('psnr/snow', value, current_iter)
                    self.snow_psnr = value
            if 'Test1' in dataset_name:
                for metric, value in self.metric_results.items():
                    tb_logger.add_scalar('psnr/rain', value, current_iter)
                    self.rain_psnr = value
            if 'Drop' in dataset_name:
                for metric, value in self.metric_results.items():
                    tb_logger.add_scalar('psnr/drop', value, current_iter)
                    self.drop_psnr = value
                    avg_psnr = (self.snow_psnr+self.rain_psnr+self.drop_psnr) / 3
                    tb_logger.add_scalar('psnr', avg_psnr, current_iter)
    
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter, best=False):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'], best=best)
        else:
            self.save_network(self.net_g, 'net_g', current_iter, best=best)
        self.save_training_state(epoch, current_iter, best=best)
