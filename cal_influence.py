import time
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import pytorch_influence_functions as ptif


def img_tensor_to_ndarray(img_tuple):
    invT = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
        )
    ])
    img = invT(img_tuple[0])
    label = img_tuple[1]
    img = np.asarray(img)
    img = img.transpose(1, 2, 0)
    # 限制图像在0, 1之间
    img = np.clip(img, 0, 1)
    return img, label


def plot_single_test_result(test_id, helpful, harmful, trainloader, testloader, classes, pred_idx):
    img, label = img_tensor_to_ndarray(testloader.dataset[test_id])
    fig, axs = plt.subplots(3, 5, sharex='col', sharey='row')
    axs[0, 0].imshow(img)
    axs[0, 0].set_title('%s(pred:%s)' % (classes[label], classes[pred_idx]))
    axs[0, 0].set_ylabel('test point(%s)' % test_id)
    for i in range(1, 5):
        axs[0, i].axis('off')
    axs[1, 0].set_ylabel('helpful')
    for i in range(5):
        img, label = img_tensor_to_ndarray(trainloader.dataset[helpful[i]])
        axs[1, i].imshow(img)
        axs[1, i].set_title(classes[label])
    axs[2, 0].set_ylabel('harmful')
    for i in range(5):
        img, label = img_tensor_to_ndarray(trainloader.dataset[harmful[i]])
        axs[2, i].imshow(img)
        axs[2, i].set_title(classes[label])
    return fig


def set_parameter_requires_grad(model, freeze_nontop):
    if freeze_nontop:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True


def cal_influence_ontest(model_path, test_point_path, train_dataset_path):
    """

    Args:
        model_path: Path of a pytorch model, .pt or .pth file.
        test_point_path: Path of a test image to be classified and explained.
        train_dataset_path: Path of the original train dataset,
                            organized like './datasets/mask', './datasets/nomask'

    Returns: Three list, test point path and predicted class, top 5 helpful train points and
            top 5 harmful train points.

    """

    gpu = 0 if torch.cuda.is_available() else -1

    timeStr = time.strftime("%Y-%m-%d_%Hh%Mm%Ss", time.localtime(time.time()))
    model = torch.load(model_path)
    set_parameter_requires_grad(model, False)
    model.eval()
    T = transforms.Compose([
            transforms.Resize([224, 224]),
            # transforms.Resize([299, 299]),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    test_set = datasets.ImageFolder(test_point_path, transform=T)
    test_loader = DataLoader(test_set, shuffle=False)
    train_set = datasets.ImageFolder(train_dataset_path, transform=T)
    train_loader = DataLoader(train_set, shuffle=False)
    classes = train_set.classes

    config = ptif.get_default_config()

    config['gpu'] = gpu
    config["recursion_depth"] =100
    config["r_averaging"] = 1
    config['damp'] = 0.1
    config['scale'] = 50000

    for test_id in range(0, len(test_loader)):
        influences, harmful, helpful, _, pred_idx = ptif.calc_influence_single(model, train_loader, test_loader,
                                                                     test_id_num=test_id,
                                                                     gpu=config["gpu"],
                                                                     recursion_depth=config["recursion_depth"],
                                                                     r=config["r_averaging"],
                                                                     damp=config['damp'],
                                                                     scale=config['scale'])
        print(influences[harmful[0]])
        print(influences[helpful[0]])
        print((test_set.imgs[test_id][0], classes[pred_idx]))
        fig = plot_single_test_result(test_id, helpful, harmful, train_loader, test_loader, classes, pred_idx)
        plt.savefig("./figs/vehicle_recogizaiton_test_%s_%s.jpg" % (test_id, timeStr))
        plt.close(fig)
        torch.cuda.empty_cache()
    # test_path = [(test_set.imgs[test_id][0], classes[test_set.imgs[test_id][1]])]
    test_path = [(test_set.imgs[test_id][0], classes[pred_idx])]
    helpful_path = []
    harmful_path = []
    for i in range(5):
        helpful_path.append((train_set.imgs[helpful[i]][0], classes[train_set.imgs[helpful[i]][1]]))
        harmful_path.append((train_set.imgs[harmful[i]][0], classes[train_set.imgs[harmful[i]][1]]))
    return test_path, helpful_path, harmful_path


parser = argparse.ArgumentParser(description='Calculate influence functions.')
parser.add_argument('--train_data_path', '-train', help='训练数据集地址, 子文件夹以类名命名, 必要参数')
parser.add_argument('--test_point_path', '-test', help='测试点地址, 请包装成与训练数据集一样的文件格式, 需要有所有类别标签的子文件夹, 其中一个包含一张测试图片即可, 必要参数')
parser.add_argument('--model_path', '-model', help='PyTorch模型地址, .pt或.pth文件, 必要参数')
# parser.add_argument('--gpu', '-g', help='是否要用GPU跑, -1表示不用, 0表示用')
args = parser.parse_args()


if __name__ == "__main__":
    # try:
    #     test_path, helpful_path, harmful_path = cal_influence_ontest(args.model_path, args.test_point_path,
    #                                                                  args.train_data_path)
    #     print(test_path, helpful_path, harmful_path)
    # except Exception as e:
    #     print(e)
    train_data_path = './car_airplane'
    test_point_path = './car_airplane_test_cover2'
    model_path = './results/resnet50_v2_Adam_2020-11-17_10h29m31s_entire.pth'

    test_path, helpful_path, harmful_path = cal_influence_ontest(model_path, test_point_path, train_data_path)

    print(test_path, helpful_path, harmful_path)