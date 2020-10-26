import copy
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim, nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import pytorch_influence_functions as ptif
from customized_model import CustomizedModel


def set_parameter_requires_grad(model, freeze_nontop):
    if freeze_nontop:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True


def cal_influence_ontest_withprob(model, test_loader, train_loader, gpu):
    """

    Args:
        model:
        test_loader:
        train_loader:
        gpu: 0用 -1不用

    Returns: helpful training points list, harmful list, (test class, pred_prob for certain class)

    """

    # timeStr = time.strftime("%Y-%m-%d_%Hh%Mm%Ss", time.localtime(time.time()))

    classes = train_loader.dataset.classes
    config = ptif.get_default_config()
    test_id = 0
    z_test, t_test = test_loader.dataset[test_id]
    z_test = test_loader.collate_fn([z_test])

    config['gpu'] = gpu
    if config['gpu'] < 0:
        model.cpu()
    else:
        z_test = z_test.cuda()

    pred = model(z_test).cpu()
    # pred_idx = pred.detach().numpy().argmax()
    pred_prob = nn.Softmax()(pred)[t_test]
    influences, harmful, helpful, _ = ptif.calc_influence_single(model, train_loader, test_loader,
                                                                 test_id_num=test_id,
                                                                 gpu=config["gpu"],
                                                                 recursion_depth=config["recursion_depth"],
                                                                 r=config["r_averaging"])
    # fig = plot_single_test_result(test_id, helpful, harmful, train_loader, test_loader, classes)
    # plt.savefig("./figs/mask_recogizaiton_test_%s_%s" % (test_id, timeStr))
    # test_path = [(test_set.imgs[test_id][0], classes[test_set.imgs[test_id][1]])]
    # test_path = [(test_set.imgs[test_id][0], classes[pred_idx])]
    # helpful_path = []
    # harmful_path = []
    # for i in range(5):
    #     helpful_path.append((train_set.imgs[helpful[i]][0], classes[train_set.imgs[helpful[i]][1]]))
    #     harmful_path.append((train_set.imgs[harmful[i]][0], classes[train_set.imgs[harmful[i]][1]]))
    return helpful, harmful, (classes[t_test], pred_prob)


def train_model(model, dataloaders_dict, num_epochs=25, is_inception=False, gpu=-1):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = torch.device("cuda:0" if gpu == 0 else "cpu")

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                              eps=1e-08, weight_decay=0, amsgrad=False)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        # print('-' * 10)

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
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
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

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        # print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def cal_prob_add_one(model_path, test_point_path, train_data_path, gpu):
    model = torch.load(model_path)
    set_parameter_requires_grad(model, False)
    model.eval()
    T = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_set = datasets.ImageFolder(test_point_path, transform=T)
    test_loader = DataLoader(test_set, shuffle=False)
    train_set = datasets.ImageFolder(train_data_path, transform=T)
    train_loader = DataLoader(train_set, shuffle=False)
    classes = train_set.classes
    # helpful, harmful, test_point_ori = cal_influence_ontest_withprob(model, test_loader, train_loader, gpu)
    # np.save("./results/helpful.npy", np.asarray(helpful))
    # np.save("./results/harmful.npy", np.asarray(harmful))
    # np.save("./results/test_point_ori.npy", np.asarray(test_point_ori))
    helpful = np.load("./results/helpful.npy")
    # harmful = np.load("./results/harmful.npy")
    # test_point_ori = np.load("./results/test_point_ori.npy")

    # 把test_loader精简成一个test point
    test_loader = DataLoader([test_set[0]], shuffle=False)

    helpful_prob = []
    for i in range(1, 21):
        model_wrapper = CustomizedModel("mobilenetv2", 2, False,
                                        use_pretrained=False)
        sub_model = model_wrapper.model
        # 每次只取影响值最大的i个train point
        sub_train_set = []
        for j in range(i):
            sub_train_set.append(train_set[helpful[j]])
        sub_train_loader = DataLoader(sub_train_set, shuffle=True)
        dataloaders_dict = {
            'train': sub_train_loader,
            'val': test_loader
        }
        print("Training on %s helpful training points..." % i)
        sub_model, _ = train_model(sub_model, dataloaders_dict, 10, gpu=gpu)

        z_test, t_test = test_loader.dataset[0]
        z_test = test_loader.collate_fn([z_test])
        if gpu < 0:
            sub_model.cpu()
        else:
            z_test = z_test.cuda()
        pred = sub_model(z_test).cpu()
        pred_prob = nn.Softmax()(pred)[t_test]
        helpful_prob.append(pred_prob)

        torch.cuda.empty_cache()

    np.save("./results/helpful_prob.npy", np.asarray(helpful_prob))

    rand_idx = [i for i in range(len(train_set))]
    random.shuffle(rand_idx)
    random_prob = []
    for i in range(1, 21):
        model_wrapper = CustomizedModel("mobilenetv2", 2, False,
                                        use_pretrained=False)
        sub_model = model_wrapper.model
        # 每次只取影响值最大的i个train point
        sub_train_set = []
        for j in range(i):
            sub_train_set.append(train_set[rand_idx[j]])
        sub_train_loader = DataLoader(sub_train_set, shuffle=True)
        dataloaders_dict = {
            'train': sub_train_loader,
            'val': test_loader
        }
        print("Training on %s random training points..." % i)
        sub_model, _ = train_model(sub_model, dataloaders_dict, 10, gpu=gpu)

        z_test, t_test = test_loader.dataset[0]
        z_test = test_loader.collate_fn([z_test])
        if gpu < 0:
            sub_model.cpu()
        else:
            z_test = z_test.cuda()
        pred = sub_model(z_test).cpu()
        pred_prob = nn.Softmax()(pred)[t_test]
        random_prob.append(pred_prob)

        torch.cuda.empty_cache()

    np.save("./results/random_prob.npy", np.asarray(random_prob))

    return helpful_prob, random_prob


if __name__ == "__main__":
    train_data_path = './mask_dataset'
    test_point_path = './mask_dataset_test'
    model_path = './results/mobilenetv2_Adam_2020-10-20_10h34m23s_entire.pth'

    # helpful_prob, random_prob = cal_prob_add_one(model_path, test_point_path, train_data_path, 0)
    helpful_prob = np.load("./results/helpful_prob.npy", allow_pickle=True)
    helpful = []
    for t in helpful_prob:
        helpful.append(t[0])
    random_prob = np.load("./results/random_prob.npy", allow_pickle=True)
    random = []
    for t in random_prob:
        random.append(t[0])

    plt.plot(helpful, label='influential')
    plt.plot(random, label='random')
    plt.xlabel("Number of training points added")
    plt.ylabel("Probability of a certain testing point")
    plt.legend()
    plt.show()

    print(helpful, random)