import cv2
import numpy as np
import os


def main():
    os.makedirs("images", exist_ok=True)
    os.makedirs("masks", exist_ok=True)

    for i in range(10):
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255  # 白背景の画像
        mask = np.zeros((256, 256), dtype=np.uint8)  # 黒背景のマスク

        if i % 2 == 0:
            center = (np.random.randint(50, 200), np.random.randint(50, 200))
            radius = np.random.randint(30, 70)
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            cv2.circle(img, center, radius, color, -1)
            cv2.circle(mask, center, radius, 255, -1)  # マスクは白
        else:
            pt1 = (np.random.randint(50, 150), np.random.randint(50, 150))
            pt2 = (np.random.randint(150, 250), np.random.randint(150, 250))
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            cv2.rectangle(img, pt1, pt2, color, -1)
            cv2.rectangle(mask, pt1, pt2, 255, -1)  # マスクは白

        cv2.imwrite(f"images/{i + 1}.png", img)
        cv2.imwrite(f"masks/{i + 1}.png", mask)


if __name__ == '__main__':
    main()
