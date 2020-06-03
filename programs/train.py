import torch
from matplotlib.axes import Axes
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm

import math

from programs import Model

DATA_DIR = '../data/npimage32.npy'
BATCH_SIZE = 128
EPOCHS = 60
LR = 0.1

CTR = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class State:
    def __init__(self):
        self.model = None
        self.epoch = 0
        self.batch = 0
        self.steps = 0
        self.writer = SummaryWriter()


def dream_image(imgs, model):
    imgs.requires_grad_(True)

    optim = torch.optim.SGD(params=[imgs],
                            momentum=0,
                            lr=0.2,
                            weight_decay=0.1)

    for i in range(20):
        y = model.encode(imgs)
        target = -y.trace()
        optim.zero_grad()
        target.backward()
        optim.step()

    imgs = imgs.detach().cpu().numpy()
    return imgs


def visualize(model):
    imgs = torch.rand(size=(model.num_units, model.size, model.size)).to(device)
    imgs = dream_image(imgs, model)

    fig, a = plt.subplots(math.ceil(model.num_units / 8), 8)

    for i, img in enumerate(imgs):
        axis: Axes = a[i // 8][i % 8]
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        axis.imshow(img, cmap='PiYG')
    plt.show()
    return imgs


def prepare_data_loader():
    data = np.load(DATA_DIR).astype('float32')
    data = torch.from_numpy(data).to(device)
    data_set = TensorDataset(data)

    data_loader = DataLoader(dataset=data_set,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             drop_last=True)
    return data_loader


def adjust_learning_rate(optimizers):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    epoch = state.epoch
    lr = LR * (0.1 ** (epoch // 40))
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def cal_loss(y1, y2, w, mask, t):
    """
    :param y1:  s_b * n_u
    :param y2:  s_b * n_u
    :param w:   s_b(q) * s_b * num_units
    :param mask: s_b * num_units
    :param t:   n_u
    """

    l_1_2 = torch.exp(-torch.abs(y1 - y2).clamp_max(5))  # s_b * n_u
    l_1_neg = torch.sum(w * torch.exp(-torch.abs(y1.unsqueeze(1) - y1).clamp_max(5)), dim=1)  # s_b * n_u
    l_2_neg = torch.sum(w * torch.exp(-torch.abs(y2.unsqueeze(1) - y1).clamp_max(5)), dim=1)

    loss = -torch.log(l_1_2 / l_1_neg) - torch.log(l_1_2 / l_2_neg)
    loss_all = loss.mean()
    loss = loss.masked_fill(~mask, 0)
    loss = loss.mean()

    state.writer.add_histogram(tag='y1', values=y1, global_step=state.steps)
    state.writer.add_histogram(tag='l12', values=l_1_2, global_step=state.steps)
    state.writer.add_histogram(tag='l1neg', values=l_1_neg, global_step=state.steps)
    state.writer.add_histogram(tag='l2neg', values=l_2_neg, global_step=state.steps)
    state.writer.add_scalar(tag='loss', scalar_value=loss_all.item(), global_step=state.steps)

    l_1_neg_ctr = torch.mean(torch.exp(-torch.abs(y1.unsqueeze(1) - y1).clamp_max(5)), dim=1)  # s_b * n_u
    l_2_neg_ctr = torch.mean(torch.exp(-torch.abs(y2.unsqueeze(1) - y1).clamp_max(5)), dim=1)

    loss_ctr = -torch.log(l_1_2 / l_1_neg_ctr) - torch.log(l_1_2 / l_2_neg_ctr)
    loss_all_ctr = loss_ctr.mean()
    loss_ctr = loss_ctr.masked_fill(~mask, 0)
    loss_ctr = loss_ctr.mean()

    state.writer.add_scalar(tag='loss_', scalar_value=(loss_all_ctr - loss_all).item(), global_step=state.steps)

    return loss_ctr if CTR else loss


def flip_grad(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            p.grad.data = - p.grad.data


def optimize(loss, optimizers):
    for optim in optimizers:
        optim.zero_grad()
    loss.backward()
    optim = optimizers[0]
    optim.step()
    optim = optimizers[1]
    flip_grad(optim)
    optim.step()


def train_epoch(model, data_loader, optimizers, ):
    for batch in data_loader:
        state.batch = batch
        (y1, w), y2, mask = model(batch[0])
        loss = cal_loss(y1, y2, w, mask, model.temperature)
        optimize(loss, optimizers)
        state.steps += 1
    pass


def train(model, data_loader, optimizers):
    visualize(model, )
    for epoch in tqdm(range(EPOCHS)):
        state.epoch = epoch
        adjust_learning_rate(optimizers)
        train_epoch(model, data_loader, optimizers, )
        # visualize(model, )
        torch.save(model.state_dict(), f='model.state_dict.' + str(epoch))


def main():
    global state
    state = State()
    model: Model = Model()
    model = model.to(device)
    state.model = model
    data_loader = prepare_data_loader()
    optimizers = [torch.optim.SGD(params=(*model.encoder.parameters(), model.temperature),
                                  lr=LR, momentum=0.9, weight_decay=0.001),
                  torch.optim.SGD(params=model.hippocampus.parameters(),
                                  lr=LR, momentum=0.9, weight_decay=0.001), ]

    train(model, data_loader, optimizers)


if __name__ == '__main__':
    main()
