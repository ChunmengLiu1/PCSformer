
import numpy as np
import torch
import os
import voc12.data
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils
import argparse
from PIL import Image
import torch.nn.functional as F
from IPython import embed
from tool import pyutils
import time

from tool.imutils import crf_inference_label

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default="network.conformer_CAM", type=str)
    parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--voc12_root", default='./VOC2012', type=str)
    parser.add_argument("--out_cam", default='save/out_cam', type=str)
    parser.add_argument("--out_crf", default='save/out_crf', type=str)
    parser.add_argument("--arch", default='sm_sub', type=str)   # round 0: sm_sub0
    parser.add_argument("--method", default='transcam', type=str)
    parser.add_argument("--save_folder", default='./save', type=str)
    parser.add_argument("--round_nb", required=True, default=None, type=int, help="the round number of the extracter")
    parser.add_argument("--k_cluster", default=10, type=int, help="the number of the sub-category")


    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.out_cam):
        os.makedirs(args.out_cam)

    model = getattr(importlib.import_module(args.network), 'Net_' + args.arch)(args.k_cluster, args.round_nb)
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ClsDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                  inter_transform=torchvision.transforms.Compose(
                                                      [
                                                       np.asarray,
                                                       imutils.Normalize(),
                                                       imutils.HWC_to_CHW]))

    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    filename_list = []
    image_feature_list = []
    timer = pyutils.Timer('extract_feature_begin:')
    start_time = time.time()
    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        img_name = img_name[0]
        filename_list.append(img_name)
        label = label[0]

        img_path = voc12.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        # extract feature
        if args.round_nb == 0:
            
            logits_conv, logits_trans, feature = model(args.method, img_list[2].cuda(), args.round_nb)
            
        else:
            logits_conv, logits_trans, feature, logits_conv_200, logits_trans_200 = model(args.method, img_list[2].cuda(), args.round_nb)

        feature = feature[0].cpu().detach().numpy()
        image_feature_list.append(feature)

        if iter % 500 == 0:
            print('Already extracted: {}/{}'.format(iter, len(infer_data_loader)))

    print("filename_length:", len(filename_list))
    image_feature_list = np.array(image_feature_list)

    print("image_feature_length:", len(image_feature_list), len(image_feature_list[0]))
    save_feature_folder_path = os.path.join(args.save_folder, 'feature')
    if not os.path.exists(save_feature_folder_path):
        os.makedirs(save_feature_folder_path)
    feature_save_path = os.path.join(save_feature_folder_path, 'R{}_feature.npy'.format(args.round_nb))
    np.save(feature_save_path, image_feature_list)
    timer = pyutils.Timer('extract_feature_end:')
    print('run_time:{} min'.format(round((time.time() - start_time) / 60, 2)))
