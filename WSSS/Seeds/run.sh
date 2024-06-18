CUDA_VISIBLE_DEVICES=1 python run_sample_dawsol.py --voc12_root "/mnt/nas/zhulei/datas/VOCdevkit/VOC2012/" \
                                    --work_space "result_voc/dawsol/" \
                                    --wsol_method "dawsol" \
                                    --rate_d 0.001 \
                                    --rate_u 0.2 \
                                    --conf_bg_thres 0.05 \
                                    --conf_fg_thres 0.3 \
                                    --sem_seg_bg_thres 0.20 \
                                    --train_cam True \
                                    --make_cam True \
                                    --eval_cam True \
                                    --cam_to_ir_label True \
                                    --train_irn True \
                                    --make_sem_seg True \
                                    --eval_sem_seg True \
                                    --seed 3

