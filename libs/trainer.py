from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import mlflow

from libs.loss import Loss
from libs.dataset import Dataset
from libs.model import Model
from libs.utils import fix_random_state, seed_worker


class Trainer(object):
    def __init__(self, cfg):
        super().__init__()
        if cfg.RANDOM_SEED is not None:
            fix_random_state(cfg.RANDOM_SEED)

        dataset = Dataset(
            cfg.DATA.ROOT_DIR,
            cfg.DATA.IMG_HEIGHT,
            cfg.DATA.IMG_WIDTH,
            image_dir_name=cfg.DATA.IMAGE_DIR,
            mask_dir_name=cfg.DATA.MASK_DIR,
            img_ext=cfg.DATA.IMG_EXT)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.DATA.BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=cfg.NUM_WORKERS,
            worker_init_fn=seed_worker)

        self.model = Model(
            cfg.MODEL.NAME,
            cfg.MODEL.ENCODER_NAME,
            classes=len(cfg.DATA.CLASS_RGB) - 1)
        self.model.to(cfg.DEVICE)
        self.model.train()

        self.loss_fn = Loss()

        self.optim = torch.optim.Adam(
                self.model.parameters(), lr=cfg.TRAIN.LR)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim,
            T_max=cfg.TRAIN.MAX_EPOCH * len(self.dataloader),
            eta_min=1e-5)

        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.device = cfg.DEVICE
        self.cfg = cfg

    def run(
            self,
            experiment_tag: str = None,
            run_name: str = None):
        if experiment_tag is not None:
            mlflow.set_experiment(experiment_name=experiment_tag)

        with mlflow.start_run(run_name=run_name) as _:
            self._log_params(self.cfg)
            self.train()

    def train(self):
        for epoch in tqdm(
                range(self.max_epoch),
                desc='Train',
                leave=False,
                dynamic_ncols=True):
            for i, inputs in tqdm(
                    enumerate(self.dataloader),
                    desc='Epoch: {}/{}'.format(epoch, self.max_epoch),
                    leave=False,
                    dynamic_ncols=True):
                iteration = len(self.dataloader) * epoch + i + 1
                outputs = self.process_batch(inputs, epoch, iteration)

        plt.imshow(inputs['image'][0].detach().numpy().transpose(1, 2, 0))
        plt.show()
        plt.imshow(outputs['mask'][0, 0].detach().numpy() > 0)
        plt.show()

    def process_batch(self, inputs, epoch, iteration):
        outputs: dict = {}

        img = self.to_device(inputs['image'])
        target = self.to_device(inputs['mask'])

        pred = self.model(img)
        outputs['mask'] = pred

        losses = self.loss_fn(pred, target)
        outputs.update(losses)
        self._log_losses(losses, iteration)
        loss = losses['loss']

        self.optim.zero_grad()
        loss.backward()

        self.optim.step()
        self.scheduler.step()

        return outputs

    def to_device(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.device, non_blocking=True)

    def _log_params(self, cfg, parent_key: str = ''):
        for key, val in cfg.items():
            key = parent_key + key
            if isinstance(val, dict):
                self._log_params(val, key + '/')
            else:
                mlflow.log_param(key, val)

    def _log_losses(self, losses: dict, iteration: int):
        for lkey, lval in losses.items():
            if isinstance(lval, torch.Tensor):
                mlflow.log_metric(
                    lkey,
                    float(lval.mean().detach().item()),
                    iteration)
