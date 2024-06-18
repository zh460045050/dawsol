
import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
import csv

def run(args):

    result_lists = []
    thresh_lists = []
    iou_lists = []
    cam_eval_thres = 0
    while cam_eval_thres < args.cam_eval_thres:
        dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)
        # labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]

        preds = []
        labels = []
        n_images = 0
        for i, id in enumerate(dataset.ids):
            n_images += 1
            cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
            cams = cam_dict['high_res']
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=cam_eval_thres)
            keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            cls_labels = np.argmax(cams, axis=0)
            cls_labels = keys[cls_labels]
            preds.append(cls_labels.copy())
            labels.append(dataset.get_example_by_keys(i, (1,))[0])

        confusion = calc_semantic_segmentation_confusion(preds, labels)

        gtj = confusion.sum(axis=1)
        resj = confusion.sum(axis=0)
        gtjresj = np.diag(confusion)
        denominator = gtj + resj - gtjresj
        iou = gtjresj / denominator

        print(args.chainer_eval_set, ", threshold:", cam_eval_thres, 'miou:', np.nanmean(iou), "i_imgs", n_images)
        print('among_predfg_bg', float((resj[1:].sum()-confusion[1:,1:].sum())/(resj[1:].sum())))
        result_lists.append(np.nanmean(iou))
        thresh_lists.append(cam_eval_thres)
        iou_lists.append(iou)
        cam_eval_thres += 0.01

    index = np.argmax(result_lists)
    print(iou_lists[index])
    with open(os.path.join(args.work_space, 'IoU_cls_%s.csv'%(args.chainer_eval_set)), "a") as f:
        writer = csv.writer(f)
        writer.writerow(iou_lists[index])

    with open(os.path.join(args.work_space, 'IoU_list_%s.csv'%(args.chainer_eval_set)), "a") as f:
        writer = csv.writer(f)
        writer.writerow(thresh_lists)
        writer.writerow(result_lists)

    max_iou = np.max(result_lists)
    print("PeakIoU: ", max_iou)
    with open(os.path.join(args.work_space.split("/")[0], 'maxIoU_%s.csv'%(args.chainer_eval_set)), "a") as f:
        writer = csv.writer(f)
        write_row = [args.work_space.split("/")[1], str(max_iou)]
        writer.writerow(write_row)

    return np.nanmean(iou)

