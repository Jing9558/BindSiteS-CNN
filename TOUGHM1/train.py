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

from dataset import TOUGHM1Pair, CacheNPY, ToMesh, ProjectOnSphere


def main(log_dir, model_path, augmentation, batch_size, learning_rate, num_workers, test_every_n, loss_margin, fold_n):
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

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

    def margin_loss(outputs, targets, margin=loss_margin):
        outputs = F.normalize(outputs[0], p=2, dim=1), F.normalize(outputs[1], p=2, dim=1)
        pw_distances = F.pairwise_distance(outputs[0], outputs[1]).view(-1)
        pos_loss = pw_distances.pow(2)
        neg_loss = torch.clamp(margin - pw_distances, min=0).pow(2)
        loss_match = torch.sum(pos_loss * targets + neg_loss * (1 - targets)) / targets.numel()
        return loss_match, pw_distances[targets > 0.5].detach(), pw_distances[targets < 0.5].detach()

    def train_step(i, j, target):
        model.train()
        i, j, target = i.cuda(), j.cuda(), target.cuda()

        i_out = model(i)
        j_out = model(j)

        loss, pos_dist, neg_dist = margin_loss((i_out, j_out), target)

        pos_dist = pos_dist.cpu().numpy().tolist()
        neg_dist = neg_dist.cpu().numpy().tolist()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), pos_dist, neg_dist

    def test_step(i, j, target):
        model.eval()
        i, j, target = i.cuda(), j.cuda(), target.cuda()

        with torch.no_grad():
            i_out = model(i)
            j_out = model(j)

        loss, pos_dist, neg_dist = margin_loss((i_out, j_out), target)

        pos_dist = pos_dist.cpu().numpy().tolist()
        neg_dist = neg_dist.cpu().numpy().tolist()

        return loss.item(), pos_dist, neg_dist

    TensorWriter = SummaryWriter(log_dir + '/tensorboard_data')
    best_test_loss = np.inf
    early_stop_counter = 0

    for epoch in range(300):
        train_set = TOUGHM1Pair("data", "train", fold_n, transform=transform)

        test_set = TOUGHM1Pair("data", "test", fold_n, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers, pin_memory=True, drop_last=True)

        max_train_samples = 25000
        max_batches = int(np.ceil(max_train_samples / batch_size))

        loss_buffer = []
        pos_dist_buffer = []
        neg_dist_buffer = []
        time_before_load = time.perf_counter()
        for batch_idx, (i, j, target) in enumerate(train_loader):
            if batch_idx + 1 >= max_batches:
                break
            time_after_load = time.perf_counter()
            time_before_step = time.perf_counter()
            loss, pos_dist, neg_dist = train_step(i, j, target)

            loss_buffer.append(loss)
            pos_dist_buffer.extend(pos_dist)
            neg_dist_buffer.extend(neg_dist)

            logger.info("[{}:{}/{}] LOSS={:.2} time={:.2}+{:.2}".format(
                epoch, batch_idx, max_batches - 1,
                loss,
                                  time_after_load - time_before_load,
                                  time.perf_counter() - time_before_step))
            time_before_load = time.perf_counter()
        total_loss = np.mean(loss_buffer)
        pos_dist = np.mean(pos_dist_buffer)
        neg_dist = np.mean(neg_dist_buffer)
        diff = np.mean(neg_dist_buffer) - np.mean(pos_dist_buffer)
        TensorWriter.add_scalar('total_loss', total_loss, epoch)
        TensorWriter.add_scalar('pos_dist', pos_dist, epoch)
        TensorWriter.add_scalar('neg_dist', neg_dist, epoch)
        TensorWriter.add_scalar('diff', diff, epoch)
        TensorWriter.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)
        scheduler.step()

        max_test_samples = 10000
        max_test_batches = int(np.ceil(max_test_samples / batch_size))

        if (epoch + 1) % test_every_n == 0:
            test_loss_buffer = []
            test_pos_dist_buffer = []
            test_neg_dist_buffer = []
            for batch_idx, (i, j, target) in enumerate(test_loader):
                if batch_idx + 1 >= max_test_batches:
                    break
                loss, pos_dist, neg_dist = test_step(i, j, target)
                test_loss_buffer.append(loss)
                test_pos_dist_buffer.extend(pos_dist)
                test_neg_dist_buffer.extend(neg_dist)
            test_total_loss = np.mean(test_loss_buffer)
            test_pos_dist = np.mean(test_pos_dist_buffer)
            test_neg_dist = np.mean(test_neg_dist_buffer)
            test_diff = test_neg_dist - test_pos_dist
            logger.info("[Test-Epoch {}] LOSS={:.2} POS_dist={:.2} NEG_dist={:.2} diff={:.2}".format(
                epoch, test_total_loss, test_pos_dist, test_neg_dist, test_diff))

            TensorWriter.add_scalar('test_total_loss', test_total_loss, epoch)
            TensorWriter.add_scalar('test_pos_dist', test_pos_dist, epoch)
            TensorWriter.add_scalar('test_neg_dist', test_neg_dist, epoch)
            TensorWriter.add_scalar('test_diff', test_diff, epoch)

            if test_total_loss <= best_test_loss:
                torch.save(model.state_dict(), os.path.join(log_dir, "state.pkl"))
                best_test_loss = test_total_loss
                early_stop_counter = 0
                logger.info("model state saved")
            else:
                early_stop_counter += 1
                print(epoch, 'early_stop_counter:', early_stop_counter)
                if early_stop_counter >= 10:
                    print("Early stopping")
                    break


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
    parser.add_argument("--test_every_n", type=int, default=1)
    parser.add_argument("--loss_margin", type=float, default=1.0)
    parser.add_argument("--fold_n", type=int, default=0)

    args = parser.parse_args()

    main(**args.__dict__)