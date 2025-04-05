from torch import nn

import segmentation_models_pytorch as smp


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = smp.losses.SoftBCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(
            smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)

        loss = bce_loss + dice_loss
        return {
            'loss': loss,
            'bce_loss': bce_loss,
            'dice_loss': dice_loss
        }
