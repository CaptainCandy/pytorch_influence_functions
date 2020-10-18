import time

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

import pytorch_influence_functions as ptif
from customized_model import load_model
from mask_recognization import processing_data


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


def plot_single_test_result(test_id, helpful, harmful, trainloader, testloader):
    # 下面的目的是不要让dataloader每次都执行一下随机裁剪
    showT = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    trainloader.dataset.change_transform(showT)
    # 下面是画图
    img, label = img_tensor_to_ndarray(testloader.dataset[test_id])
    fig, axs = plt.subplots(3, 5, sharex='col', sharey='row')
    axs[0, 0].imshow(img)
    axs[0, 0].set_title("no mask" if label == 1 else "mask")
    axs[0, 0].set_ylabel('test point(%s)' % test_id)
    for i in range(1, 5):
        axs[0, i].axis('off')
    axs[1, 0].set_ylabel('helpful')
    for i in range(5):
        img, label = img_tensor_to_ndarray(trainloader.dataset[helpful[i]])
        axs[1, i].imshow(img)
        axs[1, i].set_title("no mask" if label == 1 else "mask")
    axs[2, 0].set_ylabel('harmful')
    for i in range(5):
        img, label = img_tensor_to_ndarray(trainloader.dataset[harmful[i]])
        axs[2, i].imshow(img)
        axs[2, i].set_title("no mask" if label == 1 else "mask")
    return fig


if __name__ == "__main__":
    timeStr = time.strftime("%Y-%m-%d_%Hh%Mm%Ss", time.localtime(time.time()))
    data_dir = './datasets/5f680a696ec9b83bb0037081-momodel/data/image'
    model_name = "mobilenetv2"

    model, input_size = load_model(model_name, "./results/%s_1018_2.pth" % model_name, num_classes=2)
    # process的时候会给图片随机排序
    trainloader, testloader = processing_data(data_dir, input_size, input_size)

    config = ptif.get_default_config()
    # config["test_start_index"] = 5
    # ptif.init_logging('logfile_%s_%s.log' % (model_name, timeStr))
    # ptif.calc_img_wise(config, model, trainloader, testloader)
    test_id = 14
    influences, harmful, helpful, _ = ptif.calc_influence_single(model, trainloader, testloader,
                                                                 test_id_num=test_id,
                                                                 gpu=config["gpu"],
                                                                 recursion_depth=config["recursion_depth"],
                                                                 r=config["r_averaging"])

    fig = plot_single_test_result(test_id, helpful, harmful, trainloader, testloader)
    plt.savefig("./figs/mask_recogizaiton_test_%s" % test_id)