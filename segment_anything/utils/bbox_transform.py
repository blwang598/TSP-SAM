import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


def resize_scale_bbox(mask, expand_range=1.2):
    """
    Generate a new bounding box through random offset and scaling to accommodate uncertainty
    args:
    mask -- .png
    expand_range -- random scale e.g. 120%
    """
    # 加载mask
    mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    x, y, w, h = cv2.boundingRect(mask)

    # 生成随机扩展的宽高比例
    expand_factor_w = 1 + random.uniform(0, expand_range)
    expand_factor_h = 1 + random.uniform(0, expand_range)

    # 计算新的宽度和高度
    new_w = int(w * expand_factor_w)
    new_h = int(h * expand_factor_h)

    # 确保新的外接框完全包围原始框
    new_x = max(0, x - (new_w - w) // 2)
    new_y = max(0, y - (new_h - h) // 2)

    # 确保新的外接框不会超出图像边界
    new_w = min(new_x + new_w, mask.shape[1]) - new_x
    new_h = min(new_y + new_h, mask.shape[0]) - new_y

    # 随机偏移量范围
    offset_range_x = (new_w - w) // 2
    offset_range_y = (new_h - h) // 2

    # 生成随机偏移量，确保新的框仍能包围原始框
    random_offset_x = random.randint(-offset_range_x // 2, offset_range_x // 2)
    random_offset_y = random.randint(-offset_range_y // 2, offset_range_y // 2)

    # 应用随机偏移量
    new_x = max(0, new_x + random_offset_x)
    new_y = max(0, new_y + random_offset_y)

    # 确保新的外接框不会超出图像边界
    new_w = min(new_x + new_w, mask.shape[1]) - new_x
    new_h = min(new_y + new_h, mask.shape[0]) - new_y
    # return new_x, new_y, new_w, new_h
    # 绘制原始框和扩展后的框
    output_image = mask.copy()
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 原始框 (蓝色)
    cv2.rectangle(output_image, (new_x, new_y), (new_x + new_w, new_y + new_h), (255, 255, 255), 2)  # 随机扩展后的框 (绿色)

    # 显示结果
    plt.figure(figsize=(10, 10))
    plt.imshow(output_image)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    mask_path = "/data2/wangbilin/datasets/19-23-WLI-resize-json/0/20200619陈继刚/{1CA2AE48-9FC5-4333-8370-41A0D742003F}_mask.png"
    resize_scale_bbox(mask_path, expand_range=1)
