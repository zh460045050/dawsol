import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils
from net import resnet50


class Net(nn.Module):

    def __init__(self, stride=16, n_classes=20):
        super(Net, self).__init__()
        if stride == 16:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,self.resnet50.layer1)
        else:
            self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 1, 1), dilations=(1, 1, 2, 2))
            self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,self.resnet50.layer1)
        
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        #self.classifier = nn.Conv2d(2048, 20, 1, bias=False

        self.fc_attp_fore = nn.Conv2d(2048, 100, 1, bias=False)
        self.fc_attp_back = nn.Conv2d(2048, 100, 1, bias=False)

        self.fc8_fore = nn.Conv2d(2048, n_classes, 1, bias=False)
        self.fc8_back = nn.Conv2d(2048, n_classes, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.fc8_fore.weight)
        torch.nn.init.xavier_uniform_(self.fc8_back.weight)
        torch.nn.init.xavier_uniform_(self.fc_attp_fore.weight)
        torch.nn.init.xavier_uniform_(self.fc8_back.weight)

        self.n_classes = n_classes


        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.fc8_fore, self.fc8_back, self.fc_attp_fore, self.fc_attp_back, ])

    def forward(self, x):

        x = self.stage1(x)
        x = self.stage2(x).detach()

        x = self.stage3(x)
        x = self.stage4(x)
        
        f_pixel = x.clone()

        att_f = self.fc_attp_fore(x)
        b, c, h, w = att_f.shape
        att_f = att_f.view(b, c, h*w)
        att_f = F.softmax(att_f ,dim=2)

        att_b = self.fc_attp_back(x)
        b, c, h, w = att_b.shape
        att_b = att_b.view(b, c, h*w)
        att_b = F.softmax(att_b ,dim=2)


        #x = torchutils.gap2d(x, keepdims=True)
        #x = self.classifier(x)
        b, c, h, w = f_pixel.shape
        f_image_fore = torch.bmm(att_f, f_pixel.view(b, c, h*w).permute(0, 2, 1)).mean(dim=1).view(b, c, 1, 1)
        f_image_back = torch.bmm(att_b, f_pixel.view(b, c, h*w).permute(0, 2, 1)).mean(dim=1).view(b, c, 1, 1)

        label_fore = self.fc8_fore(f_image_fore).view(-1, self.n_classes)
        label_back = self.fc8_back(f_image_back).view(-1, self.n_classes)

        label_back_rev = self.fc8_fore(f_image_back).view(-1, self.n_classes)
        label_fore_rev = self.fc8_back(f_image_fore).view(-1, self.n_classes)

        return {"logits":label_fore, "logits_fore_rev":label_fore_rev, "logits_back":label_back, "logits_back_rev":label_back_rev}

    def train(self, mode=True):
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

        x = self.stage4(x)

        #x = F.conv2d(x, self.classifier.weight)
        x_fore = self.fc8_fore(x)
        x_fore = x_fore[0] + x_fore[1].flip(-1)

        x_back = self.fc8_back(x)
        x_back = x_back[0] + x_back[1].flip(-1)


        return x_fore #{"fore":x_fore, "back":x_back}

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
        cams = F.relu(cams)
        cams = cams/(F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5)
        cams_feature = cams.unsqueeze(2)*feature.unsqueeze(1) # bs*20*2048*32*32
        cams_feature = cams_feature.view(cams_feature.size(0),cams_feature.size(1),cams_feature.size(2),-1)
        cams_feature = torch.mean(cams_feature,-1)
        
        return x,cams_feature,cams

class CAM(Net):

    def __init__(self, stride=16,n_classes=20):
        super(CAM, self).__init__(stride=stride,n_classes=n_classes)
    
    def forward(self, x):

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        #x = F.conv2d(x, self.classifier.weight)
        x_fore = self.fc8_fore(x)
        x_fore = x_fore[0] + x_fore[1].flip(-1)

        x_back = self.fc8_back(x)
        x_back = x_back[0] + x_back[1].flip(-1)


        return x_fore #{"fore":x_fore, "back":x_back}

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
            loss_ce= F.cross_entropy(logit, label)
            loss += loss_ce
            acc += (logit.argmax(dim=1)==label.view(-1)).sum().float()
            num += label.size(0)
            
        return loss/batch_size, acc/num
