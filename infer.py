import os
from argparse import ArgumentParser
from glob import glob
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torchvision import transforms

from libs.model import SegmentationModel
from libs.utils import load_weights


def infer():
    parser = ArgumentParser()
    parser.add_argument(
        '--model', type=str,
        required=True,
        default=None,
        help='The path of the model .pth')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='The path regex of the input images.')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='The output path for the output images.')
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='The device for inference.')
    args = parser.parse_args()

    files = glob(args.input)
    checkpoint = torch.load(
        args.model, map_location=args.device, weights_only=False)

    cfg = checkpoint['config']

    class_num = \
        1 if len(cfg.DATA.CLASS_VALUES) == 2 else len(cfg.DATA.CLASS_VALUES)

    model = SegmentationModel(
        cfg.MODEL.NAME, cfg.MODEL.ENCODER_NAME, classes=class_num)
    model = load_weights(model, checkpoint['model'])
    model.to(args.device)
    model.eval()

    to_tensor = transforms.ToTensor()
    resize = transforms.Resize(
        size=(cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH))

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in tqdm(files, leave=False, dynamic_ncols=True):
        with open(filename, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        img = resize(to_tensor(img)).unsqueeze(0).to(args.device)
        with torch.inference_mode():
            logit = model(img)

        height, width = img.shape[-2:]
        img = img[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = (img * 255.0).clip(0.0, 255.0).astype(np.uint8)

        if class_num == 1:
            pred_mask = (logit.sigmoid() > 0.5).long()[0, 0]
        else:
            pred_mask = logit.softmax(dim=1).argmax(dim=1)[0]
        pred_mask = pred_mask.detach().cpu().numpy()

        fig = plt.figure(figsize=(12, 5), tight_layout=True, dpi=100)
        if height >= width:
            row = 1
            col = 2
        else:
            row = 2
            col = 1

        ax1 = fig.add_subplot(row, col, 1)
        ax1.set_axis_off()
        ax1.set_title('Input Image')

        ax1.imshow(img)

        ax2 = fig.add_subplot(row, col, 2)
        ax2.set_axis_off()
        ax2.set_title('Prediction')
        ax2.imshow(pred_mask)

        out_filename = os.path.join(
            args.output_dir, os.path.basename(filename))
        plt.savefig(out_filename)
        plt.clf()
        plt.close()


if __name__ == '__main__':
    infer()
