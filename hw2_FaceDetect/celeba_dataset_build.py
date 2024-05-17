# -*- coding: utf-8 -*-
# @Time    : 2024/5/17 9:24
# @Author  : 纪冠州
# @File    : celeba_dataset_build.py
# @Software: PyCharm 
# @Comment :

import os
import cv2
import random
import shutil
from tqdm import tqdm

celebA_Path = "./img_celeba"
celebA_bbox_path = "./list_bbox_celeba.txt"

celebA_target_path = "./yolov9-main/celeba"

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

imgs = os.listdir(celebA_Path)
random.shuffle(imgs)
imgs = random.sample(imgs, int(0.05*len(imgs)))

train_imgs = imgs[:int(len(imgs) * train_ratio)]
val_imgs = imgs[int(len(imgs) * train_ratio):int(len(imgs) * (train_ratio + val_ratio))]
test_imgs = imgs[int(len(imgs) * (train_ratio + val_ratio)):]

train_file = open(f"{celebA_target_path}/train.txt", "w")
val_file = open(f"{celebA_target_path}/val.txt", "w")
test_file = open(f"{celebA_target_path}/test.txt", "w")

with open(celebA_bbox_path, "r") as f:

    with tqdm(total=len(imgs)) as pbar:
        pbar.set_description("Processing")

        for i, line in enumerate(open(celebA_bbox_path, "r")):
            if i < 2:
                continue

            strs = line.split()
            img_name, _ = strs[0].split('.')

            x1, y1, w, h = int(strs[1]), int(strs[2]), int(strs[3]), int(strs[4])

            x2, y2 = x1 + w, y1 + h

            img = cv2.imread(f"{celebA_Path}/{strs[0]}")
            img_w, img_h = img.shape[1], img.shape[0]

            x = (x1 + w / 2.0) / img_w
            y = (y1 + h / 2.0) / img_h
            w = w / img_w
            h = h / img_h

            if x > 1 or y > 1 or w > 1 or h > 1:
                print(strs[0], " x:", x, " y:", y, " w:", w, " h:", h, " img_w:", img_w, " img_h:", img_h)

            if strs[0] in train_imgs:
                label_file = open(f"{celebA_target_path}/labels/train/{img_name}.txt", "w")
                label_file.write(f"0 {x} {y} {w} {h}\n")
                label_file.flush()
                label_file.close()

                train_file.write(f"./images/train/{strs[0]}\n")
                train_file.flush()

                shutil.copyfile(f"{celebA_Path}/{strs[0]}", f"{celebA_target_path}/images/train/{strs[0]}")
                pbar.update(1)
            elif strs[0] in val_imgs:
                label_file = open(f"{celebA_target_path}/labels/val/{img_name}.txt", "w")
                label_file.write(f"0 {x} {y} {w} {h}\n")
                label_file.flush()
                label_file.close()

                val_file.write(f"./images/val/{strs[0]}\n")
                val_file.flush()

                shutil.copyfile(f"{celebA_Path}/{strs[0]}", f"{celebA_target_path}/images/val/{strs[0]}")
                pbar.update(1)
            elif strs[0] in test_imgs:
                label_file = open(f"{celebA_target_path}/labels/test/{img_name}.txt", "w")
                label_file.write(f"0 {x} {y} {w} {h}\n")
                label_file.flush()
                label_file.close()

                test_file.write(f"./images/test/{strs[0]}\n")
                test_file.flush()

                shutil.copyfile(f"{celebA_Path}/{strs[0]}", f"{celebA_target_path}/images/test/{strs[0]}")
                pbar.update(1)
