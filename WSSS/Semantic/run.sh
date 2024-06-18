CUDA_VISIBLE_DEVICES=2 python main.py train \
    --config-path configs/voc12_dawsol.yaml

CUDA_VISIBLE_DEVICES=2 python main.py test \
    --config-path configs/voc12_dawsol.yaml \
    --model-path result/models/dawsol/deeplabv2_resnet101_msc/train_aug/checkpoint_final.pth

CUDA_VISIBLE_DEVICES=2 python main.py crf \
    --config-path configs/voc12_dawsol.yaml
