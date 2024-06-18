import argparse
import os
import numpy as np
import os.path as osp

from misc import pyutils
import torch
import random

if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    # Environment
    # parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--voc12_root", default='../VOCdevkit/VOC2012/', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")

    # Dataset
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--infer_list", default="voc12/train_aug.txt", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    parser.add_argument("--chainer_eval_set", default="train", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--feature_dim", default=2048, type=int)
    parser.add_argument("--cam_crop_size", default=512, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=10, type=int)
    parser.add_argument("--cam_learning_rate", default=0.01, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_eval_thres", default=0.5, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")

    parser.add_argument('--wsol_method', type=str, default='cam')
    # ReCAM
    parser.add_argument("--recam_num_epoches", default=4, type=int)
    parser.add_argument("--recam_learning_rate", default=0.0005, type=float)
    parser.add_argument("--recam_loss_weight", default=1.0, type=float)

    parser.add_argument("--rate_d", default=0.3, type=float)
    parser.add_argument("--rate_u", default=3.0, type=float)


    parser.add_argument('--has_grid_size', type=int, default=111)
    parser.add_argument('--has_drop_rate', type=float, default=0.19)
    parser.add_argument('--acol_threshold', type=float, default=0.57)
    parser.add_argument('--spg_threshold_1h', type=float, default=0.41,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_1l', type=float, default=0.35,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_2h', type=float, default=0.24,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_2l', type=float, default=0.21,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_3h', type=float, default=0.12,
                        help='SPG threshold')
    parser.add_argument('--spg_threshold_3l', type=float, default=0.06,
                        help='SPG threshold')
    parser.add_argument('--adl_drop_rate', type=float, default=0.59,
                        help='ADL dropout rate')
    parser.add_argument('--adl_threshold', type=float, default=0.99,
                        help='ADL gamma, threshold ratio '
                             'to maximum value of attention map')
    parser.add_argument('--cutmix_beta', type=float, default=1.35,
                        help='CutMix beta')
    parser.add_argument('--cutmix_prob', type=float, default=0.34,
                        help='CutMix Mixing Probability')

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.3, type=float)
    parser.add_argument("--conf_bg_thres", default=0.05, type=float)

    # Inter-pixel Relation Network (IRNet)
    parser.add_argument("--irn_network", default="net.resnet50_irn", type=str)
    parser.add_argument("--irn_crop_size", default=512, type=int)
    parser.add_argument("--irn_batch_size", default=32, type=int)
    parser.add_argument("--irn_num_epoches", default=3, type=int)
    parser.add_argument("--irn_learning_rate", default=0.1, type=float)
    parser.add_argument("--irn_weight_decay", default=1e-4, type=float)

    # Random Walk Params
    parser.add_argument("--beta", default=10)
    parser.add_argument("--exp_times", default=8,
                        help="Hyper-parameter that controls the number of random walk iterations,"
                             "The random walk is performed 2^{exp_times}.")
    parser.add_argument("--sem_seg_bg_thres", default=0.25)

    # Output Path
    parser.add_argument("--work_space", default="result_default5", type=str) # set your path
    parser.add_argument("--log_name", default="sample_train_eval", type=str)
    parser.add_argument("--cam_weights_name", default="res50_cam.pth", type=str)
    parser.add_argument("--irn_weights_name", default="res50_irn.pth", type=str)
    parser.add_argument("--cam_out_dir", default="cam_mask", type=str)
    parser.add_argument("--ir_label_out_dir", default="ir_label", type=str)
    parser.add_argument("--sem_seg_out_dir", default="sem_seg", type=str)
    parser.add_argument("--ins_seg_out_dir", default="ins_seg", type=str)
    parser.add_argument("--recam_weight_dir", default="recam_weight", type=str)

    # Step
    parser.add_argument("--train_cam", type=str2bool, default=True)
    parser.add_argument("--train_recam", type=str2bool, default=True)
    parser.add_argument("--make_cam", type=str2bool, default=True)
    parser.add_argument("--make_recam", type=str2bool, default=True)
    parser.add_argument("--eval_cam", type=str2bool, default=True)
    parser.add_argument("--cam_to_ir_label", type=str2bool, default=True)
    parser.add_argument("--train_irn", type=str2bool, default=True)
    parser.add_argument("--make_ins_seg", type=str2bool, default=True)
    parser.add_argument("--eval_ins_seg", type=str2bool, default=True)
    parser.add_argument("--make_sem_seg", type=str2bool, default=True) 
    parser.add_argument("--eval_sem_seg", type=str2bool, default=True)


    parser.add_argument("--seed", type=int, default=4)

    args = parser.parse_args()
    args.log_name = osp.join(args.work_space,args.log_name)
    args.cam_weights_name = osp.join(args.work_space,args.cam_weights_name)
    args.irn_weights_name = osp.join(args.work_space,args.irn_weights_name)
    args.cam_out_dir = osp.join(args.work_space,args.cam_out_dir)
    args.ir_label_out_dir = osp.join(args.work_space,args.ir_label_out_dir)
    args.sem_seg_out_dir = osp.join(args.work_space,args.sem_seg_out_dir)
    args.ins_seg_out_dir = osp.join(args.work_space,args.ins_seg_out_dir)
    args.recam_weight_dir = osp.join(args.work_space,args.recam_weight_dir)

    args.vis_cluster_dir = osp.join(args.work_space,"vis_cluster")


    os.makedirs(args.work_space, exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    os.makedirs(args.sem_seg_out_dir, exist_ok=True)
    os.makedirs(args.ins_seg_out_dir, exist_ok=True)
    os.makedirs(args.recam_weight_dir, exist_ok=True)
    os.makedirs(args.vis_cluster_dir, exist_ok=True)
    pyutils.Logger(args.log_name + '.log')
    print(vars(args))

    seed = args.seed
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    print(args.seed)

    if args.train_cam is True:
        import step.train_cam_dawsol

        timer = pyutils.Timer('step.train_cam:')
        step.train_cam_dawsol.run(args)
    

    if args.train_recam is True:
        import step.train_recam

        timer = pyutils.Timer('step.train_recam:')
        step.train_recam.run(args)
    

    if args.make_cam is True:
        import step.make_cam

        timer = pyutils.Timer('step.make_cam:')
        step.make_cam.run(args)
    
    if args.make_recam is True:
        import step.make_recam

        timer = pyutils.Timer('step.make_recam:')
        step.make_recam.run(args)

    if args.eval_cam is True:
        import step.eval_cam

        timer = pyutils.Timer('step.eval_cam:')
        step.eval_cam.run(args)
    
    
    
    if args.cam_to_ir_label is True:
        import step.cam_to_ir_label

        timer = pyutils.Timer('step.cam_to_ir_label:')
        step.cam_to_ir_label.run(args)

    if args.train_irn is True:
        import step.train_irn

        timer = pyutils.Timer('step.train_irn:')
        step.train_irn.run(args)

    if args.make_sem_seg is True:
        import step.make_sem_seg_labels
        args.sem_seg_bg_thres = float(args.sem_seg_bg_thres)
        timer = pyutils.Timer('step.make_sem_seg_labels:')
        step.make_sem_seg_labels.run(args)

    if args.eval_sem_seg is True:
        import step.eval_sem_seg

        timer = pyutils.Timer('step.eval_sem_seg:')
        step.eval_sem_seg.run(args)
    
    
    