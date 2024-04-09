set -e
set -x


# # extract feature
python extract_feature.py --weights ./save/PC_230830_10_664.pth --k_cluster 10 --round_nb 0
# # generate sub-prototype
python create_pseudo_label.py --k_cluster 10 --round_nb 0
# # train with the supervision of sub-prototype and image-level labels
python train_sp.py --weights ./weights/Conformer_small_patch16.pth --round_nb 1 --session_name SP_240408 --subcls_loss_weight 1 --k_cluster 10



# infer (optional)
python infer_sp.py --weights ./save/SP_230831_19_682.pth --round_nb 1 --k_cluster 10
python evaluation.py --predict_dir save/out_cam



# refine with psa, generate out_crf_0.60 and out_crf_0.25
python infer_sp.py \
--flag_out_crf True \
--la 0.60 \
--ha 0.25 \
--save save/ \
--weights save/SP_230831_19_682.pth \
--out_cam save/out_cam \
--out_crf save/out_crf

python train_aff.py --la_crf_dir save/out_crf_0.60 --ha_crf_dir save/out_crf_0.25

python infer_aff.py \
--weights save/resnet38_aff.pth \
--cam_dir save/out_cam \
--out_rw save/out_rw

python evaluation.py --predict_dir save/out_rw --type png



# generate final pseudo label
python CSE_voc.py


# train the segmentation network (please refer to https://github.com/kazuto1011/deeplab-pytorch)
