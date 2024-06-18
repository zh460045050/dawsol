
data_root="/mnt/nas/zhulei/datas/imagenet/" 
CUDA_VISIBLE_DEVICES=3 python main.py --data_root $data_root \
                --experiment_name ILSVRC_MMD_VGG \
                --pretrained TRUE \
                --num_val_sample_per_class 0 \
                --large_feature_map FALSE \
                --batch_size 32 \
                --epochs 10 \
                --lr 1.7E-05 \
                --lr_decay_frequency 3 \
                --weight_decay 5.00E-04 \
                --override_cache FALSE \
                --workers 16 \
                --box_v2_metric True \
                --iou_threshold_list 30 50 70 \
                --save_dir 'train_logs' \
                --seed 15 \
                --dataset_name ILSVRC \
                --architecture vgg16 \
                --wsol_method cam \
                --uda_method "mmd" \
                --beta 0.1 \
                --univer 0.3 \
                --check_path "" \
                --eval_frequency 1