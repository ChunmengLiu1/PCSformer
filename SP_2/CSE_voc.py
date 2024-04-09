import cv2
from PIL import Image
import PIL.Image
import numpy as np
import pydensecrf.densecrf as dcrf
import multiprocessing
import os
from os.path import exists
from IPython import embed

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]

cats = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv']

train_lst_path = 'voc12/trainaug_cls.txt'
sal_path = './VOC2012/saliency_map_DRS/'
att_path = 'save/attention/train_aug/'
seed_path = 'save/out_rw/'
save_path = './save/pseudo_label/'

if not exists(save_path):
    os.makedirs(save_path)

with open(train_lst_path) as f:
    lines = f.readlines()


for i, line in enumerate(lines):
    if i % 500 == 0:
        print(i)
    line = line[:-1]
    fields = line.split()
    name = fields[0]
    bg_name = sal_path + name + '.png'
    sal = cv2.imread(bg_name, 0)
    height, width = sal.shape
    gt = np.zeros((21, height, width), dtype=np.float32)
    added_gt = np.zeros((21, height, width), dtype=np.float32)
    added_gt[0] = 0.5
    sal = np.array(sal, dtype=np.float32)

    conflict = 0.9
    fg_thr = 0.4
    bg_thr = 32
    att_thr = 0.8

    gt[0] = (1 - (sal / 255))
    init_gt = np.zeros((height, width), dtype=float)
    sal_att = sal.copy()

    for i in range(len(fields) - 1):
        k = i + 1
        cls = int(fields[k])
        att_name = att_path + name + '_' + str(cls) + '.png'
        if not exists(att_name):
            continue

        att = cv2.imread(att_name, 0)
        att = (att - np.min(att)) / (np.max(att) - np.min(att) + 1e-8)
        gt[cls + 1] = att.copy()
        sal_att = np.maximum(sal_att, (att > att_thr) * 255)

    gt[gt < fg_thr] = 0

    bg = np.array(gt > conflict, dtype=np.uint8)
    bg = np.sum(bg, axis=0)
    gt = gt.argmax(0).astype(np.uint8)
    gt[bg > 1] = 255

    bg = np.array(sal_att >= bg_thr, dtype=np.uint8) * np.array(gt == 0, dtype=np.uint8)
    gt[bg > 0] = 255

    seed_name = seed_path + name + '.png'
    gt_seed = PIL.Image.open(seed_name)
    gt_seed = np.array(gt_seed)

    gt = np.where(gt==gt_seed, gt_seed, 255)

    out = gt
    valid = np.array((out > 0) & (out < 255), dtype=int).sum()
    ratio = float(valid) / float(height * width)
    if ratio < 0.01:
        out[...] = 255

    out = Image.fromarray(out.astype(np.uint8), mode='P')
    out.putpalette(palette)
    out_name = save_path + name + '.png'
    out.save(out_name)


print('over')
