from torch import nn

import segmentation_models_pytorch as smp


class SegmentationLoss(nn.Module):
    def __init__(
            self,
            ce_factor: float = 1.0,
            dice_factor: float = 1.0,
            binary_mode: bool = True):
        super().__init__()

        if binary_mode:
            self.ce = smp.losses.SoftBCEWithLogitsLoss()
            self.dice = smp.losses.DiceLoss(
                smp.losses.BINARY_MODE, from_logits=True)
        else:
            self.ce = smp.losses.SoftCrossEntropyLoss()
            self.dice = smp.losses.DiceLoss(
                smp.losses.MULTILABEL_MODE, from_logits=True)

        self.ce_factor = ce_factor
        self.dice_factor = dice_factor
        self.binary_mode = binary_mode

    def forward(self, pred, target):
        if self.binary_mode:
            target = target.to(pred.dtype)
        else:
            target = target.long()[:, 0]
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice(pred, target)

        loss = self.ce_factor * ce_loss + self.dice_factor * dice_loss
        return {
            'loss': loss,
            'ce_loss': ce_loss,
            'dice_loss': dice_loss
        }
