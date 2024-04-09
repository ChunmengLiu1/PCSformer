import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from IPython import embed
import scipy.misc
import cmapy

def save_att(label, norm_cam, save_path, img_name, nb_classes):
    for i in range(nb_classes):
        if label[i] == 1:
            att = norm_cam[i]
            att = att/(np.max(att) + 1e-8)
            att = np.array(att * 255, dtype=np.uint8)
            out_name = save_path + '/' + img_name + '_{}.png'.format(i)
            cv2.imwrite(out_name, att)
    return label

def save_att_coco(label, norm_cam, save_path, img_name, nb_classes):
    for i in range(nb_classes-1):
        if label[i] == 1:
            att = norm_cam[i]
            att = att/(np.max(att) + 1e-8)
            att = np.array(att * 255, dtype=np.uint8)   
            out_name = save_path + '/' + img_name + '_{}.png'.format(i)
            cv2.imwrite(out_name, att)
    return label


def colormap(cam, shape=None, mode=cv2.COLORMAP_JET):
    if shape is not None:
        h, w, c = shape
        cam = cv2.resize(cam, (w, h))
    cam = cv2.applyColorMap(cam,  cmapy.cmap('seismic'))
    return cam

def draw_heatmap_array(img, hm):  

    img = img.squeeze(0)
    img = img.permute(1, 2, 0) 
    hm = plt.cm.hot(hm)[:, :, :3] 

   
    hm = np.array(
        Image.fromarray((hm * 255).astype(np.uint8), 'RGB').resize((img.shape[1], img.shape[0]), Image.BICUBIC)).astype(
        np.float) * 2

   
    if hm.shape == np.array(img).astype(np.float).shape:
        
        out = (hm + np.array(img).astype(np.float)) / 3
        out = (out / np.max(out) * 255).astype(np.uint8)
    else:
        print(
            "hm.shape not equal np.array(img).astype(np.float).shape") 
    
    return hm, out  


def draw_single_heatmap(norm_cam, gt_label, orig_img, save_path, img_name):
    gt_cat = np.where(gt_label == 1)[0]
    orig_img = orig_img.squeeze(0)
    orig_img = orig_img.permute(1, 2, 0)
    orig_img = orig_img.numpy()  
    for i, gt in enumerate(gt_cat):
        
        cam_viz_path = os.path.join(save_path, img_name + '_{}.png'.format(gt))  
        show_cam_on_image(orig_img, norm_cam[gt], cam_viz_path)



def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)