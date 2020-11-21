###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
from __future__ import division
from __future__ import print_function

import copy
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from customized_model import CustomizedModel, set_parameter_requires_grad


# print("PyTorch Version: ", torch.__version__)
# print("Torchvision Version: ", torchvision.__version__)


# Print the model we just instantiated
# print(model_ft)
# 加载训练和验证数据
# 该类用于对不同的子集使用不同的transform
class DatasetFromSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

    def change_transform(self, transform):
        self.transform = transform


def load_data_normal(data_path, height=224, width=224, batch_size=32, test_split=0.3):
    """

    Args:
        data_path:
        height:
        width:
        batch_size:
        test_split: 测试集划分比例
    Returns:

    """
    trans = transforms.Compose([
                transforms.Resize(height),
                transforms.CenterCrop(height),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    dataset = datasets.ImageFolder(data_path, transform=trans)
    # 划分数据集
    train_size = int((1 - test_split) * len(dataset))
    test_size = len(dataset) - train_size
    print("Train size: %s" % train_size)
    print("Test size: %s" % test_size)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                [train_size, test_size])
    # 创建一个 DataLoader 对象
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                    shuffle=True)
    valid_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                    shuffle=True)

    return train_data_loader, valid_data_loader


def load_data(train_data_path, test_data_path, height=224, width=224, batch_size=32):
    """
    数据处理部分
    :param data_path: 数据路径
    :param height: 高度
    :param width: 宽度
    :param batch_size: 每次读取图片的数量
    :param test_split: 测试集划分比例
    :return:
    """
    trans = {
        'train': transforms.Compose([
            transforms.Resize([height, width]),
            # transforms.RandomResizedCrop(height),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize([height, width]),
            # transforms.CenterCrop(height),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = datasets.ImageFolder(train_data_path, transform=trans['train'])
    test_dataset = datasets.ImageFolder(test_data_path, transform=trans['val'])
    # 划分数据集
    # train_size = int((1 - test_split) * len(dataset))
    # test_size = len(dataset) - train_size
    print("Train size: %s" % len(train_dataset))
    print("Test size: %s" % len(test_dataset))
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset,
    #                                                             [train_size, test_size])
    # train_dataset = DatasetFromSubset(
    #     train_dataset, transform=trans['train']
    # )
    # test_dataset = DatasetFromSubset(
    #     test_dataset, transform=trans['val']
    # )
    # 创建一个 DataLoader 对象
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                    shuffle=True)
    valid_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                    shuffle=True)

    return train_data_loader, valid_data_loader


# 训练及验证
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        if model.aux_logits:
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # print('{"metric": "loss", "value": {0}, "epoch": {1}}'.format(epoch_loss, epoch))
            # print('{"metric": "accuracy", "value": {0}, "epoch": {1}}'.format(epoch_acc, epoch))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == "__main__":

    train_data_dir = './car_airplane'
    test_data_dir = './car_airplane_test'
    model_name = 'resnet50'
    num_classes = 2
    batch_size = 32
    test_split = 0.3
    num_workers = 4
    freeze_nontop = True
    num_epochs = 10
    use_pretrained = True

    # Initialize the model for this run
    model_wrapper = CustomizedModel(model_name, num_classes, freeze_nontop,
                                    use_pretrained=use_pretrained)
    model_ft = model_wrapper.model
    input_size = model_wrapper.input_size

    print("Initializing Datasets and Dataloaders...")

    train_data_loader, valid_data_loader = load_data(train_data_dir, test_data_dir,
                                                     height=input_size, width=input_size, batch_size=batch_size)
    # train_data_loader, valid_data_loader = load_data_normal(data_dir, height=input_size, width=input_size,
    #                                                         batch_size=batch_size, test_split=test_split)
    print(train_data_loader.dataset.classes)
    dataloaders_dict = {
        'train': train_data_loader,
        'val': valid_data_loader
    }

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 需要在优化器创建之前移动到计算设备上
    model_ft = model_ft.to(device)
    print('Finish loading data on %s.' % device)

    # 创建优化器
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if freeze_nontop:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name, end='')
        print()
    else:
        print('\tAll parameters.')

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.LBFGS(params_to_update, lr=1, max_iter=100, max_eval=None,
    #                            tolerance_grad=1e-07, tolerance_change=1e-09,
    #                            history_size=100, line_search_fn=None)
    # optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(params_to_update, lr=0.001, betas=(0.9, 0.999),
                              eps=1e-08, weight_decay=0, amsgrad=False)
    # optimizer_ft = optim.RMSprop(params_to_update, lr=0.01, alpha=0.99,
    #                              eps=1e-08, weight_decay=0, momentum=0, centered=False)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                                 num_epochs=num_epochs, is_inception=(model_name == "inceptionv3"))
    # 保存模型
    if freeze_nontop:
        set_parameter_requires_grad(model_ft, False)
    timeStr = time.strftime("%Y-%m-%d_%Hh%Mm%Ss", time.localtime(time.time()))
    # torch.save(model_ft.state_dict(), './results/%s_%s_%s.pth' % (model_name, "Adam", timeStr))
    torch.save(model_ft, './results/%s_v100_%s_%s_entire.pth' % (model_name, "Adam", timeStr))
    # idx = [i + 1 for i in range(len(hist))]
    # plt.plot(idx, hist)
    # plt.xticks(idx)
    plt.plot(hist)
    plt.title("Validation Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    # for a, b in zip(idx, hist):
    #     plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=14)
    plt.savefig("./figs/%s_v100_%s_%s.jpg" % (model_name, "Adam", timeStr))