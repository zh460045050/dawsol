import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50


def initialize_weights(modules, init_mode):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            if init_mode == 'he':
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif init_mode == 'xavier':
                nn.init.xavier_uniform_(m.weight.data)
            elif init_mode == 'contant':
                #print(m)
                nn.init.constant_(m.weight, 0)
            else:
                raise ValueError('Invalid init_mode {}'.format(init_mode))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
            
class AcolBase(nn.Module):
    def _acol_logits(self, feature, labels, drop_threshold):
        feat_map_a, logits = self._branch(feature=feature,
                                          classifier=self.classifier_A)
        labels = logits.argmax(dim=1).long()
        attention = get_attention(feature=feat_map_a, label=labels)
        erased_feature = _erase_attention(
            feature=feature, attention=attention, drop_threshold=drop_threshold)
        feat_map_b, logit_b = self._branch(feature=erased_feature,
                                           classifier=self.classifier_B)
        return {'logits': logits, 'logit_b': logit_b,
                'feat_map_a': feat_map_a, 'feat_map_b': feat_map_b}

    def _branch(self, feature, classifier):
        feat_map = classifier(feature)
        logits = self.avgpool(feat_map)
        logits = logits.view(logits.size(0), -1)
        return feat_map, logits

def normalize_tensor(x):
    channel_vector = x.view(x.size()[0], x.size()[1], -1)
    minimum, _ = torch.min(channel_vector, dim=-1, keepdim=True)
    maximum, _ = torch.max(channel_vector, dim=-1, keepdim=True)
    normalized_vector = torch.div(channel_vector - minimum, maximum - minimum)
    normalized_tensor = normalized_vector.view(x.size())
    return normalized_tensor

def get_attention(feature, label):
    feat_map = feature.detach().clone()
    attention = feat_map.requires_grad_(True)[
                range(feat_map.size(0)), label, :, :]
    attention = attention.unsqueeze(1)
    attention = normalize_tensor(attention)
    return attention


def _erase_attention(feature, attention, drop_threshold):
    b, _, h, w = attention.size()
    pos = torch.ge(attention, drop_threshold)
    mask = attention.new_ones((b, 1, h, w))
    mask[pos.data] = 0.
    erased_feature = feature * mask
    return erased_feature


class Net(AcolBase):

    def __init__(self, stride=16, n_classes=20):
        super(AcolBase, self).__init__()

        self.drop_threshold = 0.57

        if stride == 16:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,self.resnet50.layer1)
        else:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 1, 1), dilations=(1, 1, 2, 2))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.n_classes = n_classes

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #self.classifier = nn.Conv2d(2048, n_classes, 1, bias=False)

        self.classifier_A = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 1024, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, n_classes, 1, 1, padding=0),
        )
        self.classifier_B = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 1024, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, n_classes, 1, 1, padding=0),
        )

        initialize_weights(self.classifier_A.modules(), init_mode='xavier')
        initialize_weights(self.classifier_B.modules(), init_mode='xavier')

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier_A, self.classifier_B])


    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        feature = self.stage4(x)

        logits_dict = self._acol_logits(feature=feature, labels=None,
                                        drop_threshold=self.drop_threshold)

        return logits_dict

    def train(self, mode=True):
        super(Net, self).train(mode)
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))

class Net_CAM(Net):

    def __init__(self,stride=16,n_classes=20):
        super(Net_CAM, self).__init__(stride=stride,n_classes=n_classes)
        
    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        feature = self.stage4(x)

        logits_dict = self._acol_logits(feature=feature, labels=None,
                                drop_threshold=self.drop_threshold)

        cams = torch.max(logits_dict["feat_map_a"], logits_dict["feat_map_b"])
        cams = F.relu(cams)

        return logits_dict["logits"], cams, feature

class Net_CAM_Feature(Net):

    def __init__(self,stride=16,n_classes=20):
        super(Net_CAM_Feature, self).__init__(stride=stride,n_classes=n_classes)
        
    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x)

        x = self.stage3(x)
        feature = self.stage4(x) # bs*2048*32*32

        x = torchutils.gap2d(feature, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, self.n_classes)

        cams = F.conv2d(feature, self.classifier.weight)
        #cams = F.relu(cams)
        
        #cams = torch.nn.LeakyReLU(0.01)(cams)

        #print(F.adaptive_max_pool2d(cams, (1, 1)))

        cams = cams/(F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5)
        cams_feature = cams.unsqueeze(2)*feature.unsqueeze(1) # bs*20*2048*32*32
        cams_feature = cams_feature.view(cams_feature.size(0),cams_feature.size(1),cams_feature.size(2),-1)
        cams_feature = torch.mean(cams_feature,-1)
        
        return x,cams_feature,cams

class CAM(Net):

    def __init__(self, stride=16,n_classes=20):
        super(CAM, self).__init__(stride=stride,n_classes=n_classes)

    def forward(self, x, separate=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        feature = self.stage4(x)

        logits_dict = self._acol_logits(feature=feature, labels=None,
                                drop_threshold=self.drop_threshold)

        x = torch.max(logits_dict["feat_map_a"], logits_dict["feat_map_b"])
        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x

    def forward1(self, x, weight, separate=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.conv2d(x, weight)
        
        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)

        return x

    def forward2(self, x, weight, separate=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.conv2d(x, weight*self.classifier.weight)
        
        if separate:
            return x
        x = F.relu(x)
        x = x[0] + x[1].flip(-1)
        return x

class Class_Predictor(nn.Module):
    def __init__(self, num_classes, representation_size):
        super(Class_Predictor, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Conv2d(representation_size, num_classes, 1, bias=False)

    def forward(self, x, label):
        batch_size = x.shape[0]
        x = x.reshape(batch_size,self.num_classes,-1) # bs*20*2048
        mask = label>0 # bs*20

        feature_list = [x[i][mask[i]] for i in range(batch_size)] # bs*n*2048
        prediction = [self.classifier(y.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1) for y in feature_list]
        labels = [torch.nonzero(label[i]).squeeze(1) for i in range(label.shape[0])]

        loss = 0
        acc = 0
        num = 0
        for logit,label in zip(prediction, labels):
            if label.shape[0] == 0:
                continue
            loss_ce= F.cross_entropy(logit + 1e-10, label)
            #loss_ce= F.cross_entropy(logit, label)
            loss += loss_ce
            acc += (logit.argmax(dim=1)==label.view(-1)).sum().float()
            num += label.size(0)
        
        return loss/batch_size, acc/num

