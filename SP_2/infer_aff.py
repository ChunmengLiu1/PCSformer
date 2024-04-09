import torch
import torchvision
from tool import imutils

import argparse
import importlib
import numpy as np

import voc12.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os.path
from PIL import Image

from tool.imutils import crf_inference_label

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]

def get_indices_in_radius(height, width, radius):

    search_dist = []
    for x in range(1, radius):
        search_dist.append((0, x))

    for y in range(1, radius):
        for x in range(-radius+1, radius):
            if x*x + y*y < radius*radius:
                search_dist.append((y, x))

    full_indices = np.reshape(np.arange(0, height * width, dtype=np.int64),
                              (height, width))
    radius_floor = radius-1
    cropped_height = height - radius_floor
    cropped_width = width - 2 * radius_floor

    indices_from = np.reshape(full_indices[:-radius_floor, radius_floor:-radius_floor], [-1])

    indices_from_to_list = []

    for dy, dx in search_dist:

        indices_to = full_indices[dy:dy + cropped_height, radius_floor + dx:radius_floor + dx + cropped_width]
        indices_to = np.reshape(indices_to, [-1])

        indices_from_to = np.stack((indices_from, indices_to), axis=1)

        indices_from_to_list.append(indices_from_to)

    concat_indices_from_to = np.concatenate(indices_from_to_list, axis=0)

    return concat_indices_from_to


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default='save/resnet38_aff.pth')
    parser.add_argument("--network", default="network.resnet38_aff", type=str)
    # parser.add_argument("--infer_list", default="voc12/train.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--cam_dir", type=str, default='save/out_cam')
    parser.add_argument("--voc12_root", default='./VOC2012', type=str)
    parser.add_argument("--bkg", default=0.46, type=float)
    parser.add_argument("--out_rw", default='save/out_rw', type=str)
    parser.add_argument("--beta", default=14, type=int)
    parser.add_argument("--logt", default=7, type=int)
    parser.add_argument("--crf", default=True, type=bool)

    args = parser.parse_args()

    if not os.path.exists(args.out_rw):
        os.makedirs(args.out_rw)

    model = getattr(importlib.import_module(args.network), 'Net')()

    model.load_state_dict(torch.load(args.weights), strict=False)

    model.eval()
    model.cuda()

    infer_dataset = voc12.data.VOC12ImageDataset(args.infer_list, voc12_root=args.voc12_root,
                                               transform=torchvision.transforms.Compose(
        [np.asarray,
         imutils.Normalize(),
         imutils.HWC_to_CHW]))
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    for iter, (name, img) in enumerate(infer_data_loader):

        name = name[0]
        if iter % 500 == 0:
            print(iter)

        orig_shape = img.shape
        padded_size = (int(np.ceil(img.shape[2]/8)*8), int(np.ceil(img.shape[3]/8)*8))

        p2d = (0, padded_size[1] - img.shape[3], 0, padded_size[0] - img.shape[2])
        img = F.pad(img, p2d)

        dheight = int(np.ceil(img.shape[2]/8))
        dwidth = int(np.ceil(img.shape[3]/8))

        cam = np.load(os.path.join(args.cam_dir, name + '.npy'), allow_pickle=True).item()

        cam_full_arr = np.zeros((21, orig_shape[2], orig_shape[3]), np.float32)
        for k, v in cam.items():
            cam_full_arr[k+1] = v

        cam_full_arr[0] = args.bkg
        cam_full_arr = np.pad(cam_full_arr, ((0, 0), (0, p2d[3]), (0, p2d[1])), mode='constant')

        with torch.no_grad():
            aff_mat = torch.pow(model.forward(img.cuda(), True), args.beta)

            trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
            for _ in range(args.logt):
                trans_mat = torch.matmul(trans_mat, trans_mat)

            cam_full_arr = torch.from_numpy(cam_full_arr)
            cam_full_arr = F.avg_pool2d(cam_full_arr, 8, 8)
            cam_vec = cam_full_arr.view(21, -1)

            cam_rw = torch.matmul(cam_vec.cuda(), trans_mat)
            cam_rw = cam_rw.view(1, 21, dheight, dwidth)
       
            cam_rw = torch.nn.Upsample((img.shape[2], img.shape[3]), mode='bilinear')(cam_rw)

            if args.crf:
                img_ori = img[0].numpy().transpose((1, 2, 0))
                img_ori = np.ascontiguousarray(img_ori)
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
                img_ori[:, :, 0] = (img_ori[:, :, 0] * std[0] + mean[0]) * 255
                img_ori[:, :, 1] = (img_ori[:, :, 1] * std[1] + mean[1]) * 255
                img_ori[:, :, 2] = (img_ori[:, :, 2] * std[2] + mean[2]) * 255
                img_ori[img_ori > 255] = 255
                img_ori[img_ori < 0] = 0
                img_ori = img_ori.astype(np.uint8)

                predict = np.argmax(cam_rw[0].cpu().numpy(), axis=0).astype(np.uint8)
                crf = crf_inference_label(img_ori, predict)
                pred = Image.fromarray(crf.astype(np.uint8)[:orig_shape[2], :orig_shape[3]], mode='P')
                pred.putpalette(palette)
                pred.save(os.path.join(args.out_rw, name + '.png'))
                # Image.fromarray(crf.astype(np.uint8)[:orig_shape[2], :orig_shape[3]]).save(
                #     os.path.join(args.out_rw, name + '.png'))
            else:
                _, cam_rw_pred = torch.max(cam_rw, 1)
                res = np.uint8(cam_rw_pred.cpu().data[0])[:orig_shape[2], :orig_shape[3]]
                pred = Image.fromarray(res, mode='P')
                pred.putpalette(palette)
                pred.save(os.path.join(args.out_rw, name + '.png'))

                # Image.fromarray(res).save(os.path.join(args.out_rw, name + '.png'))

# pred = np.asarray(np.argmax(np.concatenate((bg_score, norm_cam)), 0), dtype=int)
#             pred = Image.fromarray(pred.astype(np.uint8), mode='P')
#             pred.putpalette(palette)
#             pred.save(os.path.join(args.out_cam_pred, img_name + '.png'))
