import torch
from torch import nn

import segmentation_models_pytorch as smp


class SegmentationModel(nn.Module):
    def __init__(
            self,
            model_name: str,
            encoder_name: str,
            classes: int = 1):
        super().__init__()

        self.model = getattr(smp, model_name)(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1)

        preprocess_params = smp.encoders.get_preprocessing_params(
            encoder_name, pretrained='imagenet')
        self.mean = nn.Parameter(torch.tensor(
            preprocess_params['mean'],
            dtype=torch.float32).view(1, -1, 1, 1))
        self.std = nn.Parameter(torch.tensor(
            preprocess_params['std'],
            dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x):
        """
        Return:
            logits
        """
        x = (x - self.mean) / self.std
        y = self.model(x)
        return y
