import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import timm
from tool.mixstyle import MixStyle

######################################################################


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        # For old pytorch, you may use kaiming_normal.
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, linear=512, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear > 0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(linear)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num=751, droprate=0.5, stride=2, circle=False, ibn=False, linear_num=512,
                 model_subtype="50", mixstyle=True):
        super(ft_net, self).__init__()
        if model_subtype in ("50", "default"):
            if ibn:
                model_ft = torch.hub.load(
                    'XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
            else:
                model_ft = models.resnet50(weights="IMAGENET1K_V2")
        elif model_subtype == "101":
            if ibn:
                model_ft = torch.hub.load("XingangPan/IBN-Net", "resnet101_ibn_a", pretrained=True)
            else:
                model_ft = models.resnet101(weights="IMAGENET1K_V2")
        elif model_subtype == "152":
            if ibn:
                raise ValueError("Resnet152 has no IBN variants available.")
            model_ft = models.resnet152(weights="IMAGENET1K_V2")
        else:
            raise ValueError(f"Resnet model subtype: {model_subtype} is invalid, choose from: ['50','101','152'].")
        
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.circle = circle
        self.classifier = ClassBlock(
            2048, class_num, droprate, linear=linear_num, return_f=circle)
        self.mixstyle = MixStyle(alpha=0.3) if mixstyle else None

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        if self.training and self.mixstyle:
            x = self.mixstyle(x)
        x = self.model.layer2(x)
        if self.training and self.mixstyle:
            x = self.mixstyle(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the swin_base_patch4_window7_224 Model
class ft_net_swin(nn.Module):
    def __init__(self, class_num, droprate=0.5, circle=False, linear_num=512, **kwargs):
        super(ft_net_swin, self).__init__()
        model_ft = timm.create_model(
            'swin_base_patch4_window7_224', pretrained=True, drop_rate=droprate)
        model_ft.head = nn.Sequential()
        self.model = model_ft
        self.circle = circle
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = ClassBlock(
            1024, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.avgpool(x.permute(0, 2, 1))
        x = x.squeeze(2)
        x = self.classifier(x)
        return x

    
# Define the HRNet18-based Model
class ft_net_hr(nn.Module):
    def __init__(self, class_num, droprate=0.5, circle=False, linear_num=512):
        super().__init__()
        model_ft = timm.create_model('hrnet_w18', pretrained=True)
        # avg pooling to global pooling
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.classifier = nn.Sequential()  # save memory
        self.model = model_ft
        self.circle = circle
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = ClassBlock(
            2048, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False, linear_num=512):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.circle = circle
        # For DenseNet, the feature dim is 1024
        self.classifier = ClassBlock(
            1024, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the Efficient-based Model
class ft_net_efficient(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False, linear_num=512, model_subtype="b4"):
        super().__init__()
        try:
            from efficientnet_pytorch import EfficientNet
        except ImportError:
            print('Please pip install efficientnet_pytorch')

        if model_subtype == "default":
            model_subtype = "b4"
        subtype_int = int(model_subtype[-1])

        # For EfficientNet, the feature dim is not fixed
        out_channels_by_type = [1280, 1280, 1408,
                                1536, 1792, 2048, 2304, 2560, 2816]

        model_ft = EfficientNet.from_pretrained(
            'efficientnet-{}'.format(model_subtype))

        model_ft.head = nn.Sequential()
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.classifier = nn.Sequential()
        self.model = model_ft
        self.circle = circle

        self.classifier = ClassBlock(
            out_channels_by_type[subtype_int], class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        #x = self.model.forward_features(x)
        x = self.model.extract_features(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x


# Define the NAS-based Model
class ft_net_NAS(nn.Module):

    def __init__(self, class_num, droprate=0.5, circle=False, linear_num=512):
        super().__init__()

        # hotfix for expired certificate of https://data.lip6.fr/
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        
        model_ft = timm.create_model("nasnetalarge", pretrained=True,
                                     drop_rate=droprate)
        model_ft.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        self.classifier = ClassBlock(
            4032, class_num, droprate, linear=linear_num, return_f=circle)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num=751, droprate=0.5):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x)  # use our classifier.
        return x

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num):
        super(PCB, self).__init__()

        self.part = 6  # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(2048, class_num,
                                           droprate=0.5, linear=256, relu=False, bnorm=True))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = x[:, :, i].view(x.size(0), x.size(1))
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])

        # sum prediction
        #y = predict[0]
        # for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y


class PCB_test(nn.Module):
    def __init__(self, model):
        super(PCB_test, self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0), x.size(1), x.size(2))
        return y


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
    # Here I left a simple forward function.
    # Test the model, before you train it.
    net = ft_net_swin(751)
    # net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(2, 3, 224, 224))
    output = net(input)
    print('net output size:')
    print(output.shape)
