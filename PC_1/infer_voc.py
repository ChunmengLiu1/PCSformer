import numpy as np
import torch
import os
import voc12.data
import importlib
from torch.utils.data import DataLoader
import torchvision
from tool import imutils, infer_utils
import argparse
from PIL import Image
import torch.nn.functional as F
from IPython import embed
from tool import pyutils
from tool.imutils import crf_inference_label

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default="network.conformer_CAM", type=str)
    parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    # parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--voc12_root", default='./VOC2012', type=str)
    parser.add_argument("--out_cam", default='save/out_cam', type=str)
    parser.add_argument("--out_crf", default='save/out_crf', type=str)
    parser.add_argument("--arch", default='sm21', type=str)
    parser.add_argument("--method", default='transcam', type=str)
    parser.add_argument("--save", default='./save', type=str)
    parser.add_argument("--nb_classes", default=20, type=int)


    args = parser.parse_args()
    print(args)
    
    attention_folder = args.save + '/attention'
    if not os.path.exists(attention_folder):
        os.makedirs(attention_folder)

    if not os.path.exists(args.out_cam):
        os.makedirs(args.out_cam)

    heatmap_folder = args.save + '/heatmap'
    if not os.path.exists(heatmap_folder):
        os.makedirs(heatmap_folder)

    model = getattr(importlib.import_module(args.network), 'Net_' + args.arch)()
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ClsDatasetMSF_origscale(args.infer_list, voc12_root=args.voc12_root,
                                                      scales=(0.25, 1, 1.5, 2),
                                                      inter_transform=torchvision.transforms.Compose(
                                                          [np.asarray,
                                                              imutils.Normalize(),
                                                              imutils.HWC_to_CHW]))


    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    timer = pyutils.Timer('infer begin:')
    for iter, (img_name, img_list, label) in enumerate(infer_data_loader):
        if iter % 500 == 0:
            print('iter:', iter)
        img_name = img_name[0]
        label = label[0]

        img_path = voc12.data.get_img_path(img_name, args.voc12_root)
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        cam_list = []

        with torch.no_grad():
            for i, img in enumerate(img_list):
                logits_conv, logits_trans, cam = model(args.method, img.cuda())

                cam = F.interpolate(cam[:, 1:, :, :], orig_img_size, mode='bilinear', align_corners=False)[0]
                cam = cam.cpu().numpy() * label.clone().view(args.nb_classes, 1, 1).numpy()

                if i % 2 == 1:
                    cam = np.flip(cam, axis=-1)
                cam_list.append(cam)

        sum_cam = np.sum(cam_list, axis=0)
        sum_cam[sum_cam < 0] = 0
        cam_max = np.max(sum_cam, (1, 2), keepdims=True)
        cam_min = np.min(sum_cam, (1, 2), keepdims=True)
        sum_cam[sum_cam < cam_min + 1e-5] = 0
        norm_cam = (sum_cam - cam_min - 1e-5) / (cam_max - cam_min + 1e-5)
        
        ZERO = infer_utils.save_att(label, norm_cam, attention_folder, img_name, args.nb_classes)

        orig_img_ht = torch.from_numpy(orig_img)
        orig_img_ht = orig_img_ht.permute(2, 0, 1)
        orig_img_ht = orig_img_ht.unsqueeze(0)
        infer_utils.draw_single_heatmap(norm_cam, label, orig_img_ht, heatmap_folder, img_name)

        cam_dict = {}
        for i in range(args.nb_classes):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        if args.out_cam is not None:
            np.save(os.path.join(args.out_cam, img_name + '.npy'), cam_dict)

    timer = pyutils.Timer('infer end:')
