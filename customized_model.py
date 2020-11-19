import os
import sys

import torch
import torch.nn as nn
from torchvision import models


# 这一步是准备要使用的模型，以及更改模型的顶层分类器
# 冻结预训练好的权重
def set_parameter_requires_grad(model, freeze_nontop):
    if freeze_nontop:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True


def get_pretrain_model_url(model_name):
    models = [
        'alexnet',
        'densenet121', 'densenet169', 'densenet201', 'densenet161',
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'inceptionv3', 'squeezenet1_0', 'squeezenet1_1',
        'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
        'vgg19_bn', 'vgg19', 'mobilenetv2'
    ]
    assert model_name in models, "Model not supported."

    model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
        'densenet121': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet121-fbdb23505.pth',
        'densenet169': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet169-f470b90a4.pth',
        'densenet201': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet201-5750cbb1e.pth',
        'densenet161': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet161-347e6b360.pth',
        'inceptionv3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
        'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
        'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
        'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
        'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
        'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
        'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
        'mobilenetv2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
    }
    return model_urls[model_name]


def manully_download_pretrain_params(model_name):
    """

    Args:
        model_name: str, name of the vision model. Currently support vgg16, vgg16_bn, vgg19, vgg19_bn,
            resnet50, alexnet.

    Returns:
        Path of pretrained params.

    """

    model_url = get_pretrain_model_url(model_name)
    # 下载pretrain的参数到特定的位置
    torch.utils.model_zoo.load_url(model_url,
                                   model_dir='./models', map_location=None, progress=True, check_hash=False)
    path = './models/' + model_url.split('/')[-1]
    return path


def get_pretrain_model_path(model_name):
    model_url = get_pretrain_model_url(model_name)
    path = './models/' + model_url.split('/')[-1]
    return path


def load_model(model_name, model_path, num_classes):
    if model_name == "vgg16":
        model_ft = models.vgg16(pretrained=False)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg16_bn":
        model_ft = models.vgg16_bn(pretrained=False)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg19":
        model_ft = models.vgg19(pretrained=False)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg19_bn":
        model_ft = models.vgg19_bn(pretrained=False)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=False)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        model_ft = models.resnet50(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "mobilenetv2":
        model_ft = models.mobilenet_v2(pretrained=False)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        sys.exit()

    # 加载训练好的模型
    state_dict = torch.load(model_path)
    model_ft.load_state_dict(state_dict)
    model_ft.cuda()

    return model_ft, input_size


class CustomizedModel:
    model = None
    input_size = 0

    def __init__(self, model_name, num_classes, freeze_nontop=True, use_pretrained=False):
        """

        Args:
            model_name: str, "vgg16" or "vgg16_bn" or "vgg19" or "vgg19_bn" or "resnet50" or "alexnet" or "mobilenetv2"
            num_classes: int,
            freeze_nontop: bool,
            use_pretrained: bool,
            params_path: str, The path that saved the params of a certain model.
        """

        if model_name == "resnet50":
            self.model = models.resnet50(pretrained=False)
            if use_pretrained:
                # 先判断下模型预训练的参数有没有下载好
                params_path = get_pretrain_model_path(model_name)
                if not os.path.isfile(params_path):
                    params_path = manully_download_pretrain_params(model_name)
                # 把预训练的参数load出来，再用初始化的模型读取
                state_dict = torch.load(params_path)
                self.model.load_state_dict(state_dict)
                set_parameter_requires_grad(self.model, freeze_nontop)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            self.input_size = 224

        elif model_name == "alexnet":
            self.model = models.alexnet(pretrained=False)
            if use_pretrained:
                # 先判断下模型预训练的参数有没有下载好
                params_path = get_pretrain_model_path(model_name)
                if not os.path.isfile(params_path):
                    params_path = manully_download_pretrain_params(model_name)
                # 把预训练的参数load出来，再用初始化的模型读取
                state_dict = torch.load(params_path)
                self.model.load_state_dict(state_dict)
                set_parameter_requires_grad(self.model, freeze_nontop)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
            self.input_size = 224

        elif model_name == "vgg16":
            self.model = models.vgg16(pretrained=False)
            if use_pretrained:
                # 先判断下模型预训练的参数有没有下载好
                params_path = get_pretrain_model_path(model_name)
                if not os.path.isfile(params_path):
                    params_path = manully_download_pretrain_params(model_name)
                # 把预训练的参数load出来，再用初始化的模型读取
                state_dict = torch.load(params_path)
                self.model.load_state_dict(state_dict)
                set_parameter_requires_grad(self.model, freeze_nontop)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
            self.input_size = 224

        elif model_name == "vgg16_bn":
            self.model = models.vgg16_bn(pretrained=False)
            if use_pretrained:
                # 先判断下模型预训练的参数有没有下载好
                params_path = get_pretrain_model_path(model_name)
                if not os.path.isfile(params_path):
                    params_path = manully_download_pretrain_params(model_name)
                # 把预训练的参数load出来，再用初始化的模型读取
                state_dict = torch.load(params_path)
                self.model.load_state_dict(state_dict)
                set_parameter_requires_grad(self.model, freeze_nontop)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
            self.input_size = 224

        elif model_name == "vgg19":
            self.model = models.vgg19(pretrained=False)
            if use_pretrained:
                # 先判断下模型预训练的参数有没有下载好
                params_path = get_pretrain_model_path(model_name)
                if not os.path.isfile(params_path):
                    params_path = manully_download_pretrain_params(model_name)
                # 把预训练的参数load出来，再用初始化的模型读取
                state_dict = torch.load(params_path)
                self.model.load_state_dict(state_dict)
                set_parameter_requires_grad(self.model, freeze_nontop)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
            self.input_size = 224

        elif model_name == "vgg19_bn":
            self.model = models.vgg16_bn(pretrained=False)
            if use_pretrained:
                # 先判断下模型预训练的参数有没有下载好
                params_path = get_pretrain_model_path(model_name)
                if not os.path.isfile(params_path):
                    params_path = manully_download_pretrain_params(model_name)
                # 把预训练的参数load出来，再用初始化的模型读取
                state_dict = torch.load(params_path)
                self.model.load_state_dict(state_dict)
                set_parameter_requires_grad(self.model, freeze_nontop)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)
            self.input_size = 224

        elif model_name == "mobilenetv2":
            self.model = models.mobilenet_v2(pretrained=False)
            if use_pretrained:
                # 先判断下模型预训练的参数有没有下载好
                params_path = get_pretrain_model_path(model_name)
                if not os.path.isfile(params_path):
                    params_path = manully_download_pretrain_params(model_name)
                # 把预训练的参数load出来，再用初始化的模型读取
                state_dict = torch.load(params_path)
                self.model.load_state_dict(state_dict)
                set_parameter_requires_grad(self.model, freeze_nontop)
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            self.input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            self.model = models.squeezenet1_0(pretrained=False)
            if use_pretrained:
                # 先判断下模型预训练的参数有没有下载好
                params_path = get_pretrain_model_path(model_name)
                if not os.path.isfile(params_path):
                    params_path = manully_download_pretrain_params(model_name)
                # 把预训练的参数load出来，再用初始化的模型读取
                state_dict = torch.load(params_path)
                self.model.load_state_dict(state_dict)
                set_parameter_requires_grad(self.model, freeze_nontop)
            self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            self.model.num_classes = num_classes
            self.input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            self.model = models.densenet121(pretrained=False)
            if use_pretrained:
                # 先判断下模型预训练的参数有没有下载好
                params_path = get_pretrain_model_path(model_name)
                if not os.path.isfile(params_path):
                    params_path = manully_download_pretrain_params(model_name)
                # 把预训练的参数load出来，再用初始化的模型读取
                state_dict = torch.load(params_path)
                self.model.load_state_dict(state_dict)
                set_parameter_requires_grad(self.model, freeze_nontop)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, num_classes)
            self.input_size = 224

        elif model_name == "inceptionv3":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            self.model = models.inception_v3(pretrained=use_pretrained, aux_logits=False)
            set_parameter_requires_grad(self.model, freeze_nontop)
            # if use_pretrained:
            #     # 先判断下模型预训练的参数有没有下载好
            #     params_path = get_pretrain_model_path(model_name)
            #     if not os.path.isfile(params_path):
            #         params_path = manully_download_pretrain_params(model_name)
            #     # 把预训练的参数load出来，再用初始化的模型读取
            #     state_dict = torch.load(params_path)
            #     self.model.load_state_dict(state_dict)
            #     set_parameter_requires_grad(self.model, freeze_nontop)
            # Handle the auxilary net
            # num_ftrs = self.model.AuxLogits.fc.in_features
            # self.model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            self.input_size = 299

        else:
            print("Invalid model name, exiting...")
            sys.exit()