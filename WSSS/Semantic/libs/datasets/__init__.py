from .voc import VOC, VOCAug, VOC_WS
from .cocostuff import CocoStuff10k, CocoStuff164k


def get_dataset(name):
    return {
        "cocostuff10k": CocoStuff10k,
        "cocostuff164k": CocoStuff164k,
        "voc": VOC,
        "vocaug": VOCAug,
        "vocws": VOC_WS
    }[name]
