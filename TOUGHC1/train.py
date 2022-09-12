import torch
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter

import os
import shutil
import time
import logging
import copy
import types
import importlib.machinery
import numpy as np
from torch.optim.lr_scheduler import StepLR

from dataset import TOUGHC1, CacheNPY, ToMesh, ProjectOnSphere


def main(log_dir, model_path, augmentation, batch_size, learning_rate, num_workers):
    arguments = copy.deepcopy(locals())

    os.mkdir(log_dir)
    shutil.copy2(__file__, os.path.join(log_dir, "script.py"))
    shutil.copy2(model_path, os.path.join(log_dir, "model.py"))

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_dir, "log.txt"))
    logger.addHandler(fh)

    logger.info("%s", repr(arguments))

    torch.backends.cudnn.benchmark = True

    # Load the model
    loader = importlib.machinery.SourceFileLoader('model', os.path.join(log_dir, "model.py"))
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)

    model = mod.DeepSphereCls(3072)
    model.cuda()

    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    nside = 16

    # Load the dataset
    # Increasing `repeat` will generate more cached files
    transform = CacheNPY(prefix="b{}_".format(nside), repeat=augmentation, transform=torchvision.transforms.Compose(
        [
            ToMesh(random_rotations=True, random_translation=0.1),
            ProjectOnSphere(nside=nside)
        ]
    ))

    def target_transform(x):
        classes = ['heme', 'nucleotide', 'control']
        return classes.index(x)

    train_set = TOUGHC1("data", transform=transform, target_transform=target_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                               pin_memory=True, drop_last=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

    def train_step(data, target):
        model.train()
        data, target = data.cuda(), target.cuda()

        prediction = model(data)
        loss = F.nll_loss(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = prediction.data.max(1)[1].eq(target.data).long().cpu().sum()

        return loss.item(), correct.item()

    def test_step(data, target):
        model.eval()
        data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            prediction = model(data)
        loss = F.nll_loss(prediction, target)

        test_pred = prediction.data.max(1)[1]
        test_pred_score = prediction.data.cpu().numpy()
        test_pred_score = np.exp(test_pred_score)
        np.set_printoptions(suppress=True)
        test_true = target.data

        correct = prediction.data.max(1)[1].eq(target.data).long().cpu().sum()

        return loss.item(), correct.item(), test_pred, test_true, test_pred_score

    TensorWriter = SummaryWriter(log_dir + '/tensorboard_data')

    for epoch in range(30):

        total_loss = 0
        total_correct = 0
        time_before_load = time.perf_counter()
        for batch_idx, (data, target) in enumerate(train_loader):
            time_after_load = time.perf_counter()
            time_before_step = time.perf_counter()
            loss, correct = train_step(data, target)

            total_loss += loss
            total_correct += correct

            logger.info("[{}:{}/{}] LOSS={:.2} <LOSS>={:.2} ACC={:.2} <ACC>={:.2} time={:.2}+{:.2}".format(
                epoch, batch_idx, len(train_loader),
                loss, total_loss / (batch_idx + 1),
                      correct / len(data), total_correct / len(data) / (batch_idx + 1),
                      time_after_load - time_before_load,
                      time.perf_counter() - time_before_step))
            time_before_load = time.perf_counter()
        train_acc = total_correct / len(train_loader) / train_loader.batch_size
        TensorWriter.add_scalar('train_ACC', train_acc, epoch)
        TensorWriter.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)
        scheduler.step()

        torch.save(model.state_dict(), os.path.join(log_dir, "state.pkl"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--augmentation", type=int, default=1,
                        help="Generate multiple image with random rotations and translations")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.5)

    args = parser.parse_args()

    main(**args.__dict__)