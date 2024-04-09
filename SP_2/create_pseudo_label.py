import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.manifold import TSNE

import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle

import scipy.misc as sim
from PIL import Image
import pandas as pd
import shutil
import os
import argparse
import time
from IPython import embed
from tool import pyutils

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def read_train_list():
    with open('./voc12/train_aug.txt', 'r') as r:
        filename_list = r.read().split('\n')
    r.close()

    filename_list = filename_list[:-1]
    for num, filename in enumerate(filename_list):
        filename_list[num] = filename[12:23]
    return filename_list



def make_subclass_label(class_idx, labels, subclass_nb, all_class_kmeans_label):
    final_label = []

    for i in range(len(labels)):
        label = [0]*subclass_nb*20
        label[class_idx*subclass_nb+labels[i]] = 1
        all_class_kmeans_label.append(label)
       
    return all_class_kmeans_label


def generate_merged_label(repeat_list, label_list):
   
    new_label_list = []
    for num, one in enumerate(repeat_list):
        
        merged_label = []
        for i in one:
            merged_label.append(label_list[i])

        merged_label = sum(np.array(merged_label))

        nb_subcategory = np.nonzero(merged_label)[0].shape[0]

        if len(one) == nb_subcategory:
            new_label_list.append(merged_label)

    return new_label_list


def create_class_key_in_dict(dict, cls_nb):
    for i in range(cls_nb):
        dict[i] = []

    return dict



def make_filename_class_dict(filenamelist, labels):
    filename_idx_class_dict = {}
    filename_idx_class_dict = create_class_key_in_dict(filename_idx_class_dict, 20)


    filename_class_dict = {}
    filename_class_dict = create_class_key_in_dict(filename_class_dict, 20)

    for num, one in enumerate(labels):
        gt_labels = np.where(one == 1)[0]
        for gt in gt_labels:
            filename_idx_class_dict[gt].append(num)
            filename_class_dict[gt].append(filenamelist[num])

    return filename_idx_class_dict, filename_class_dict



def merge_filename_class_dict(filename_class_dict):
    merged_filename_list = []
    for i in range(20):
        class_filename_list = filename_class_dict[i]
        for one in class_filename_list:
            merged_filename_list.append(one)
    return merged_filename_list


def generate_repeat_list(filename_list):
    repeat_list = []
    for num, one in enumerate(filename_list):
        
        repeat_idx_list = []
        for num_, one_ in enumerate(filename_list):
            if one == one_:
                repeat_idx_list.append(num_)

        repeat_list.append(repeat_idx_list)
        
    return repeat_list


def remove_duplicate_label(repeat_list):
   
    keep_idx_list = []
    remove_idx_list = []
    for repeat_set in repeat_list:
        for num, one in enumerate(repeat_set):
            if num == 0 and one not in keep_idx_list:
                keep_idx_list.append(one)
               
            else:
                remove_idx_list.append(one)
               
    return keep_idx_list, remove_idx_list



def create_train_data(merge_filename_list, new_label_list, keep_idx_list):
    label_20 = np.load('./voc12/20_class_labels.npy')

    train_filename_list = []
    train_label_200 = []
    train_label_20 = []
    for idx in keep_idx_list:
        
        train_filename_list.append(merge_filename_list[idx])
        train_label_200.append(new_label_list[idx])
        train_label_20.append(label_20[idx])
       

        tmp = np.where(new_label_list[idx] == 1)[0]
        cls200_par = [int(x/10) for x in tmp]
    return train_filename_list, train_label_200, train_label_20




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_cluster", required=True, type=int, help="the number of the sub-category")
    parser.add_argument("--round_nb", required=True, type=int, help="the round number")
    parser.add_argument("--save_folder", default='./save', type=str, help="the path to save the sub-category label")
    parser.add_argument("--save_label", default='./save/label', type=str, help="folder")
    

    args = parser.parse_args()

    if not os.path.exists(args.save_label):
        os.makedirs(args.save_label)

    timer = pyutils.Timer('create_pseudo_label_begin:')
    start_time = time.time()
    feature_folder_path = os.path.join(args.save_folder, 'feature')

    filenamelist = read_train_list()
    cls20_label = np.load('./voc12/train_label.npy')

    filename_idx_class_dict, filename_class_dict = make_filename_class_dict(filenamelist, cls20_label)
    
    merge_filename_list = merge_filename_class_dict(filename_class_dict)
    repeat_list = generate_repeat_list(merge_filename_list)
    keep_idx_list, remove_idx_list = remove_duplicate_label(repeat_list)

    features = np.load('{}/R{}_feature.npy'.format(feature_folder_path, args.round_nb))

    print(len(filenamelist), cls20_label.shape, features.shape)


    all_class_kmeans_label = []
    for i in range(20):
        filename_idx_list = filename_idx_class_dict[i]

        class_feature_list = []
        for idx in filename_idx_list:
            class_feature_list.append(features[idx])

        print('Class {}: {}'.format(i, len(class_feature_list)))

        # apply kmeans
        X = class_feature_list
        k_cluster = args.k_cluster
        max_iter = 300
        k_center = KMeans(n_clusters=k_cluster, random_state=0, max_iter=max_iter).fit(X)
        labels = k_center.labels_
        centers = k_center.cluster_centers_
        iter_nb = k_center.n_iter_
        distance = k_center.inertia_
        subclass_nb = k_cluster
        all_class_kmeans_label = make_subclass_label(i, labels, subclass_nb, all_class_kmeans_label)

    new_label_list = generate_merged_label(repeat_list, all_class_kmeans_label)

    train_filename_list, train_label_200, train_label_20 = create_train_data(merge_filename_list, new_label_list, keep_idx_list)
    
    train_label_200 = np.array(train_label_200)
    train_label_20 = np.array(train_label_20)
    


    with open('{}/label/R{}_train_filename_list.txt'.format(args.save_folder, args.round_nb), 'w') as f:
        for x in train_filename_list:
            line = '{}\n'.format(x)
            f.write(line)
    f.close()
   
    np.save('{}/label/R{}_train_label_200.npy'.format(args.save_folder, args.round_nb), train_label_200)
    np.save('{}/label/R{}_train_label_20.npy'.format(args.save_folder, args.round_nb), train_label_20)
    

    print('===== k_{} Round-{} pseudo labels are saved at {}/label. ====='.format(args.k_cluster, args.round_nb, args.save_folder))
    print('===== You can start to train the classification model. =====')
    timer = pyutils.Timer('create_pseudo_label_begin:')
    print('run_time:{} min'.format(round((time.time()-start_time)/60, 2)))
