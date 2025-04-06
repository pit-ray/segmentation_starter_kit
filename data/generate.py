import os

import numpy as np
import cv2


def generate_toy_dataset(num_trains: int, num_vals: int):
    for phase, num in [('train', num_trains), ('val', num_vals)]:
        image_dir = os.path.join('toy_dataset', phase, 'images')
        mask_dir = os.path.join('toy_dataset', phase, 'masks')
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        for i in range(num):
            img = np.ones((256, 256, 3), dtype=np.uint8) * 255
            mask = np.zeros((256, 256), dtype=np.uint8)

            if i % 2 == 0:
                center = (
                    np.random.randint(50, 200),
                    np.random.randint(50, 200)
                )
                radius = np.random.randint(30, 70)
                color = tuple(np.random.randint(0, 255, size=3).tolist())
                cv2.circle(img, center, radius, color, -1)
                cv2.circle(mask, center, radius, 255, -1)
            else:
                pt1 = (
                    np.random.randint(50, 150),
                    np.random.randint(50, 150)
                )
                pt2 = (
                    np.random.randint(150, 250),
                    np.random.randint(150, 250)
                )
                color = tuple(np.random.randint(0, 255, size=3).tolist())
                cv2.rectangle(img, pt1, pt2, color, -1)
                cv2.rectangle(mask, pt1, pt2, 255, -1)

            cv2.imwrite(os.path.join(image_dir, '{}.png'.format(i + 1)), img)
            cv2.imwrite(os.path.join(mask_dir, '{}.png'.format(i + 1)), mask)


if __name__ == '__main__':
    generate_toy_dataset(100, 10)
