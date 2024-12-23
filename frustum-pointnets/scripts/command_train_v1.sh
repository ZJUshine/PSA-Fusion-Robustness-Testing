#/bin/bash
# 预训练模型训练脚本
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir train/log_v1 --num_point 1024 --max_epoch 201 --batch_size 32 --decay_step 800000 --decay_rate 0.5 --restore_model_path train/log_v1/model.ckpt
# 从头训练脚本
python train/train.py --gpu 0 --model frustum_pointnets_v1 --log_dir train/log_v1 --num_point 1024 --max_epoch 201 --batch_size 32 --decay_step 800000 --decay_rate 0.5
