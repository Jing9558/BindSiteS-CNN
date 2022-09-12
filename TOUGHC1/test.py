import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics import confusion_matrix

import os
import logging
import copy
import types
import importlib.machinery
import numpy as np

from dataset import TOUGHC1_steroid, CacheNPY, ToMesh, ProjectOnSphere

def main(log_dir, model_path, augmentation, batch_size, num_workers):
    arguments = copy.deepcopy(locals())

    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_dir, "test_log.txt"))
    logger.addHandler(fh)

    logger.info("%s", repr(arguments))

    torch.backends.cudnn.benchmark = True

    # Load the model
    loader = importlib.machinery.SourceFileLoader('model', os.path.join(log_dir, "model.py"))
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)

    model = mod.DeepSphereCls(3072)
    model.cuda()

    model.load_state_dict(torch.load(os.path.join(log_dir, "state.pkl")), strict=False)

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

    val_set = TOUGHC1_steroid("data", transform=transform, target_transform=target_transform)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                             pin_memory=True, drop_last=True)

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

    for epoch in range(1):

        test_total_loss = 0
        test_total_correct = 0
        test_preds = []
        test_trues = []
        test_heme_score = []
        test_nucleotide_score = []
        test_control_score = []
        for batch_idx, (data, target) in enumerate(val_loader):
            loss, correct, test_pred, test_true, test_pred_score = test_step(data, target)
            test_total_loss += loss
            test_total_correct += correct
            test_pred = test_pred.cpu().numpy()
            test_preds += test_pred.tolist()
            test_true = test_true.cpu().numpy()
            test_trues += test_true.tolist()
            test_heme_score += test_pred_score[:, 0].tolist()
            test_nucleotide_score += test_pred_score[:, 1].tolist()
            test_control_score += test_pred_score[:, 2].tolist()
        test_loss = test_total_loss / val_loader.batch_size
        test_acc = test_total_correct / len(val_loader) / val_loader.batch_size
        logger.info("[{}/{}] <LOSS>={:.2} <ACC>={:.2} ".format(
            epoch, 'test',
            test_loss,
            test_acc,
        ))
        cnf = confusion_matrix(test_trues, test_preds)
        logger.info('test confusion matrix:')
        logger.info(cnf)
        logger.info("test_trues:")
        logger.info(test_trues)
        logger.info("control_score:")
        logger.info(test_control_score)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--augmentation", type=int, default=1,
                        help="Generate multiple image with random rotations and translations")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()

    main(**args.__dict__)