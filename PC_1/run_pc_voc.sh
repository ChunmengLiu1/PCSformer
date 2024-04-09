set -e
set -x

# train
python train_voc.py --weights ./weights/Conformer_small_patch16.pth --session_name PC_240408

# infer
python infer_voc.py --weights ./save/PC_230830_10_664.pth
python evaluation.py --predict_dir save/out_cam
