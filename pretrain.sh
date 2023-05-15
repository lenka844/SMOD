CUDA_VISIBLE_DEVICES=2 python DDP_simsiam_ccrop_pretrain.py configs/small/underwater/simsiam_ccrop_pretrain.py
python self-weight_converter_pre.py
cd ../mmdetection
bash train.sh