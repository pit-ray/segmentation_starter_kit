from collections import defaultdict
import contextlib

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils import data
import mlflow
import segmentation_models_pytorch as smp

from libs.loss import SegmentationLoss
from libs.dataset import SegmentationDataset
from libs.model import SegmentationModel
from libs.utils import fix_random_state, seed_worker


class Trainer(object):
    def __init__(self, cfg):
        super().__init__()
        if cfg.RANDOM_SEED is not None:
            fix_random_state(cfg.RANDOM_SEED)

        if len(cfg.DATA.CLASS_VALUES) == 2:
            # Binary class without background class.
            self.class_num = 1
            self.binary_mode = True
            print('Binary Mode')
        else:
            # Multi-class including the background class.
            self.class_num = len(cfg.DATA.CLASS_VALUES)
            self.binary_mode = False
            print('Multi-label Mode')

        self.dataloaders = self._init_dataloaders(cfg)
        self.model = self._init_model(cfg)
        self.loss_fn = self._init_loss(cfg)
        self.optim, self.scheduler = self._init_optimizer(cfg)

        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.device = cfg.DEVICE
        self.cfg = cfg

    def _init_dataloaders(self, cfg):
        datasets = {
            'train': SegmentationDataset(
                        cfg.DATA.TRAIN_ROOT_DIR,
                        size_hw=(cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH),
                        class_values=cfg.DATA.CLASS_VALUES,
                        image_dir_name=cfg.DATA.IMAGE_DIR,
                        mask_dir_name=cfg.DATA.MASK_DIR,
                        img_ext=cfg.DATA.IMG_EXT,
                        is_train=True,
                        random_hflip=True,
                        random_vflip=True,
                        random_colorjit=True,
                        random_crop=False),
            'val': SegmentationDataset(
                        cfg.DATA.VAL_ROOT_DIR,
                        size_hw=(cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH),
                        class_values=cfg.DATA.CLASS_VALUES,
                        image_dir_name=cfg.DATA.IMAGE_DIR,
                        mask_dir_name=cfg.DATA.MASK_DIR,
                        img_ext=cfg.DATA.IMG_EXT,
                        is_train=False)
        }
        dataloaders = {
            'train': data.DataLoader(
                        datasets['train'],
                        batch_size=cfg.DATA.BATCH_SIZE,
                        shuffle=True,
                        pin_memory=True,
                        drop_last=True,
                        num_workers=cfg.NUM_WORKERS,
                        worker_init_fn=seed_worker),
            'val': data.DataLoader(
                        datasets['val'],
                        batch_size=1,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False,
                        num_workers=cfg.NUM_WORKERS,
                        worker_init_fn=seed_worker)
        }
        return dataloaders

    def _init_model(self, cfg):
        model = SegmentationModel(
            cfg.MODEL.NAME,
            cfg.MODEL.ENCODER_NAME,
            classes=self.class_num)
        model.to(cfg.DEVICE)
        return model

    def _init_loss(self, cfg):
        loss_fn = SegmentationLoss(
            ce_factor=cfg.LOSS.CE_FACTOR,
            dice_factor=cfg.LOSS.DICE_FACTOR,
            binary_mode=self.binary_mode)
        loss_fn.to(cfg.DEVICE)
        return loss_fn

    def _init_optimizer(self, cfg):
        optim = torch.optim.AdamW(
                self.model.parameters(), lr=cfg.TRAIN.LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optim,
            total_iters=cfg.TRAIN.MAX_EPOCH * len(self.dataloaders['train']),
            power=0.9)
        return optim, scheduler

    def run(
            self,
            experiment_tag: str = None,
            run_name: str = None):
        if experiment_tag is not None:
            mlflow.set_experiment(experiment_name=experiment_tag)

        with mlflow.start_run(run_name=run_name) as _:
            self._log_params(self.cfg)

            for epoch in range(self.max_epoch):
                self.train(epoch)
                self.val(epoch)

                checkpoints = {
                    'epoch': epoch + 1,
                    'config': self.cfg,
                    'model': self.model.state_dict(),
                    'optimizer': self.optim.state_dict(),
                    'loss': self.loss_fn.state_dict(),
                    'schaduler': self.scheduler.state_dict(),
                    'last_epoch': self.scheduler.last_epoch
                }
                mlflow.pytorch.log_state_dict(
                    checkpoints, 'final_checkpoint')

    def train(self, epoch):
        self.model.train()

        with tqdm(
                self.dataloaders['train'],
                desc='Train {}/{}'.format(epoch + 1, self.max_epoch),
                leave=False,
                dynamic_ncols=True) as pbar:
            for i, inputs in enumerate(pbar):
                iteration = len(self.dataloaders['train']) * epoch + i + 1
                outputs, losses = self.process_batch(inputs)

                scalar_losses = {}
                for lkey, lval in losses.items():
                    if isinstance(lval, torch.Tensor):
                        lval = float(lval.mean().detach().item())
                        scalar_losses[lkey] = lval
                pbar.set_postfix(scalar_losses)

                self.optim.zero_grad()
                losses['loss'].backward()

                self.optim.step()
                self.scheduler.step()

                for group_id, group in enumerate(self.optim.param_groups):
                    mlflow.log_metric(
                        'lr{}'.format(group_id), group['lr'], iteration)

        for key, val in scalar_losses.items():
            mlflow.log_metric('train/' + key, val, iteration)

        with self._save_figure(inputs, outputs) as fig:
            mlflow.log_figure(fig, 'train/{:05}.jpg'.format(epoch))

    def val(self, epoch):
        self.model.eval()

        iou_scores = []
        val_losses = defaultdict(float)
        iteration = len(self.dataloaders['train']) * (epoch + 1)
        with tqdm(
                self.dataloaders['val'],
                desc='Val {}/{}'.format(epoch + 1, self.max_epoch),
                leave=False,
                dynamic_ncols=True) as pbar:
            for i, inputs in enumerate(pbar):
                with torch.inference_mode():
                    outputs, losses = self.process_batch(inputs)

                for lkey, lval in losses.items():
                    if isinstance(lval, torch.Tensor):
                        val_losses[lkey] = float(lval.mean().detach().item())

                    logit = outputs['logit']
                    if self.binary_mode:
                        pred_mask = logit.sigmoid() > 0.5
                        metric_mode = 'binary'
                    else:
                        pred_mask = logit.softmax(dim=1).argmax(dim=1)
                        metric_mode = 'multiclass'
                    target_mask = inputs['mask']
                    tp, fp, fn, tn = smp.metrics.get_stats(
                            pred_mask, target_mask,
                            mode=metric_mode, num_classes=self.class_num)
                    iou_score = smp.metrics.iou_score(
                        tp, fp, fn, tn, reduction="micro")
                    iou_scores.append(iou_score)

        mlflow.log_metric(
            'val/iou_score', np.array(iou_score).mean(), iteration)

        for key, val in val_losses.items():
            mlflow.log_metric('val/' + key, val, iteration)

        with self._save_figure(inputs, outputs) as fig:
            mlflow.log_figure(fig, 'val/{:05}.jpg'.format(epoch))

    def process_batch(self, inputs: dict) -> tuple:
        outputs: dict = {}

        img = self.to_device(inputs['image'])
        target = self.to_device(inputs['mask'])

        pred = self.model(img)
        outputs['logit'] = pred

        losses = self.loss_fn(pred, target)

        return outputs, losses

    def to_device(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.device, non_blocking=True)

    def _log_params(self, cfg, parent_key: str = ''):
        for key, val in cfg.items():
            key = parent_key + key
            if isinstance(val, dict):
                self._log_params(val, key + '/')
            else:
                mlflow.log_param(key, val)

    @contextlib.contextmanager
    def _save_figure(self, inputs, outputs) -> plt.Figure:
        img = inputs['image'][0].detach().cpu().numpy()
        height, width = img.shape[-2:]
        img = img.transpose(1, 2, 0)
        img = (img * 255.0).clip(0.0, 255.0).astype(np.uint8)

        pred = outputs['logit']
        if self.binary_mode:
            pred_mask = (pred.sigmoid() > 0.5).long()[0, 0]
        else:
            pred_mask = pred.softmax(dim=1).argmax(dim=1)[0]
        pred_mask = pred_mask.detach().cpu().numpy()

        target_mask = inputs['mask'][0, 0].detach().cpu().numpy()

        fig = plt.figure(figsize=(12, 5), tight_layout=True, dpi=100)

        if height >= width:
            row = 1
            col = 3
        else:
            row = 3
            col = 1

        ax1 = fig.add_subplot(row, col, 1)
        ax1.set_axis_off()
        ax1.set_title('Input Image')
        ax1.imshow(img)

        ax2 = fig.add_subplot(row, col, 2)
        ax2.set_axis_off()
        ax2.set_title('Prediction')
        ax2.imshow(pred_mask)

        ax3 = fig.add_subplot(row, col, 3)
        ax3.set_axis_off()
        ax3.set_title('Ground Truth')
        ax3.imshow(target_mask)

        try:
            yield fig
        finally:
            plt.clf()
            plt.close()
