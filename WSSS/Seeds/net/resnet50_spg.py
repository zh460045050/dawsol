import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50
from torch.utils.model_zoo import load_url

_ADL_POSITION = [[], [], [], [0], [0, 2]]
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'inception_v3_google':
        'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

def get_attention(feature, label):
    feat_map = feature.detach().clone()
    attention = feat_map.requires_grad_(True)[
                range(feat_map.size(0)), label, :, :]
    attention = attention.unsqueeze(1)
    attention = normalize_tensor(attention)
    return attention

def compute_attention(feat_map, labels, logits_b1, logits_b2):
    upsample_module = nn.Upsample(size=(224, 224), mode='bilinear')
    attention = get_attention(upsample_module(feat_map), labels)
    fused_attention = _get_fused_attention(logits_b1, logits_b2)
    return attention, fused_attention

def _get_fused_attention(feature1, feature2):
    upsample_module = nn.Upsample(size=(224, 224), mode='bilinear')
    feat_map1 = feature1.detach().clone()
    feat_map2 = feature2.detach().clone()
    return (torch.sigmoid(upsample_module(feat_map1)) +
            torch.sigmoid(upsample_module(feat_map2))) / 2.



def align_layer(state_dict):
    keys = [key for key in sorted(state_dict.keys())]
    for key in reversed(keys):
        move = 0
        if 'layer' not in key:
            continue
        key_sp = key.split('.')
        layer_idx = int(key_sp[0][-1])
        block_idx = key_sp[1]
        if not _ADL_POSITION[layer_idx]:
            continue

        for pos in reversed(_ADL_POSITION[layer_idx]):
            if pos < int(block_idx):
                move += 1

        key_sp[1] = str(int(block_idx) + move)
        new_key = '.'.join(key_sp)
        state_dict[new_key] = state_dict.pop(key)
    return state_dict


def batch_replace_layer(state_dict):
    state_dict = replace_layer(state_dict, 'layer3.0.', 'SPG_A1.0.')
    state_dict = replace_layer(state_dict, 'layer3.1.', 'SPG_A2.0.')
    state_dict = replace_layer(state_dict, 'layer3.2.', 'SPG_A2.1.')
    state_dict = replace_layer(state_dict, 'layer3.3.', 'SPG_A2.2.')
    state_dict = replace_layer(state_dict, 'layer3.4.', 'SPG_A2.3.')
    state_dict = replace_layer(state_dict, 'layer3.5.', 'SPG_A2.4.')
    return state_dict


def normalize_tensor(x):
    channel_vector = x.view(x.size()[0], x.size()[1], -1)
    minimum, _ = torch.min(channel_vector, dim=-1, keepdim=True)
    maximum, _ = torch.max(channel_vector, dim=-1, keepdim=True)
    normalized_vector = torch.div(channel_vector - minimum, maximum - minimum)
    normalized_tensor = normalized_vector.view(x.size())
    return normalized_tensor


def remove_layer(state_dict, keyword):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict

def replace_layer(state_dict, keyword1, keyword2):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword1 in key:
            new_key = key.replace(keyword1, keyword2)
            state_dict[new_key] = state_dict.pop(key)
            if key == 'fc.weight':
                state_dict[new_key] = state_dict[new_key].unsqueeze(-1).unsqueeze(-1)
    return state_dict


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


def load_pretrained_model(model, wsol_method, path=None, **kwargs):
    strict_rule = True

    if path:
        state_dict = torch.load(os.path.join(path, 'resnet50.pth'))
    else:
        state_dict = load_url(model_urls['resnet50'], progress=True)

    if wsol_method == 'adl':
        state_dict = align_layer(state_dict)
        state_dict = remove_layer(state_dict, 'fc')
        strict_rule = False
    elif wsol_method == 'spg':
        state_dict = batch_replace_layer(state_dict)
        state_dict = remove_layer(state_dict, 'fc')
    else:
        state_dict = remove_layer(state_dict, 'fc')
        strict_rule = False


    model.load_state_dict(state_dict, strict=strict_rule)
    return model

def get_downsampling_layer(inplanes, block, planes, stride):
    outplanes = planes * block.expansion
    if stride == 1 and inplanes == outplanes:
        return
    else:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )


class Bottleneck(nn.Module):
    
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetSpg(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 large_feature_map=False, **kwargs):
        super(ResNetSpg, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block=block, planes=64,
                                       blocks=layers[0],
                                       stride=1, split=False)
        self.layer2 = self._make_layer(block=block, planes=128,
                                       blocks=layers[1],
                                       stride=2, split=False)
        self.SPG_A1, self.SPG_A2 = self._make_layer(block=block, planes=256,
                                                    blocks=layers[2],
                                                    stride=2,
                                                    split=True)
        self.layer4 = self._make_layer(block=block, planes=512,
                                       blocks=layers[3],
                                       stride=1, split=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        initialize_weights(self.modules(), init_mode='xavier')

    def _make_layer(self, block, planes, blocks, stride, split=None):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        first_layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        other_layers = []
        for _ in range(1, blocks):
            other_layers.append(block(self.inplanes, planes))

        if split:
            return nn.Sequential(*first_layers), nn.Sequential(*other_layers)
        else:
            return nn.Sequential(*(first_layers + other_layers))

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers


def resnet50_spg(architecture_type, pretrained=False, pretrained_path=None,
             **kwargs):
    model =  ResNetSpg(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model = load_pretrained_model(model, architecture_type,
                                      path=pretrained_path, **kwargs)
    return model


class Net(nn.Module):

    def __init__(self, stride=16, n_classes=20):
        super(Net, self).__init__()
        if stride == 16:
            self.resnet50 = resnet50_spg("spg", pretrained=True)
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,self.resnet50.layer1)
        else:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 1, 1), dilations=(1, 1, 2, 2))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3_1 = nn.Sequential(self.resnet50.SPG_A1)
        self.stage3_2 = nn.Sequential(self.resnet50.SPG_A2)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.n_classes = n_classes

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.SPG_A4 = nn.Conv2d(2048, n_classes,
                        kernel_size=1)

        self.SPG_B_1a = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_B_2a = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_B_shared = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1),
        )

        self.SPG_C = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
        )

        initialize_weights(self.SPG_C.modules(), init_mode='xavier')
        initialize_weights(self.SPG_B_shared.modules(), init_mode='xavier')
        initialize_weights(self.SPG_B_2a.modules(), init_mode='xavier')
        initialize_weights(self.SPG_B_1a.modules(), init_mode='xavier')
        initialize_weights(self.SPG_A4.modules(), init_mode='xavier')


        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3_1, self.stage3_2, self.stage4])
        self.newly_added = nn.ModuleList([self.SPG_A4, self.SPG_B_1a, self.SPG_B_2a, self.SPG_B_shared, self.SPG_C ])


    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3_1(x)

        logits_b1 = self.SPG_B_1a(x)
        logits_b1 = self.SPG_B_shared(logits_b1)

        x = self.stage3_2(x)
        logits_b2 = self.SPG_B_2a(x)
        logits_b2 = self.SPG_B_shared(logits_b2)

        x = self.stage4(x)

        feat_map = self.SPG_A4(x)

        logits_c = self.SPG_C(x)

        logits = self.avgpool(feat_map)
        logits = logits.view(logits.shape[0:2])

        labels = logits.argmax(dim=1).long()
        attention, fused_attention = compute_attention(
            feat_map=feat_map, labels=labels,
            logits_b1=logits_b1, logits_b2=logits_b2)

        return {'attention': attention, 'fused_attention': fused_attention,
                        'logits': logits, 'logits_b1': logits_b1,
                        'logits_b2': logits_b2, 'logits_c': logits_c}

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
        x = self.stage3_1(x)

        logits_b1 = self.SPG_B_1a(x)
        logits_b1 = self.SPG_B_shared(logits_b1)

        x = self.stage3_2(x)
        logits_b2 = self.SPG_B_2a(x)
        logits_b2 = self.SPG_B_shared(logits_b2)

        x = self.stage4(x)

        feat_map = self.SPG_A4(x)

        logits_c = self.SPG_C(x)

        logits = self.avgpool(feat_map)
        logits = logits.view(logits.shape[0:2])

        labels = logits.argmax(dim=1).long()
        attention, fused_attention = compute_attention(
            feat_map=feat_map, labels=labels,
            logits_b1=logits_b1, logits_b2=logits_b2)

        cams = feat_map.clone().detach()
        cams = torch.relu(cams)
        
        return logits,cams,x

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
        x = self.stage3_1(x)

        logits_b1 = self.SPG_B_1a(x)
        logits_b1 = self.SPG_B_shared(logits_b1)

        x = self.stage3_2(x)
        logits_b2 = self.SPG_B_2a(x)
        logits_b2 = self.SPG_B_shared(logits_b2)

        x = self.stage4(x)

        feat_map = self.SPG_A4(x)
        
        logits = self.avgpool(feat_map)
        logits = logits.view(logits.shape[0:2])

        x = feat_map.clone().detach()
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

