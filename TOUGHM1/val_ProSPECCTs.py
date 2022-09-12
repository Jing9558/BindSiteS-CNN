import torch
import torch.nn.functional as F
import torchvision

import os
import shutil
import time
import logging
import copy
import types
import importlib.machinery
import numpy as np
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import roc_curve, auc,average_precision_score

from dataset import TOUGHM1Pair, CacheNPY, ToMesh, ProjectOnSphere, TOUGHM1Pair_test, ProSPECCTsPairs


def main(log_dir, model_path, augmentation, batch_size, num_workers, db_name):
    arguments = copy.deepcopy(locals())

    logger = logging.getLogger("val")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_dir, "PS_val_log.txt"))
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

    test_set = ProSPECCTsPairs("data", db_name, transform=transform)

    def sim_cal(outputs):
        outputs = F.normalize(outputs[0], p=2, dim=1), F.normalize(outputs[1], p=2, dim=1)
        pw_distances = F.pairwise_distance(outputs[0], outputs[1]).view(-1)
        sim = -pw_distances
        return sim

    def test_step(i,j, target):
        model.eval()
        i,j,target = i.cuda(), j.cuda(), target.cuda()

        with torch.no_grad():
            i_out = model(i)
            j_out = model(j)

        sim = sim_cal((i_out,j_out))

        sim=sim.cpu().numpy().tolist()
        label=target.data.cpu().numpy().tolist()

        return sim,label

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)

    sims = []
    labels = []

    for batch_idx, (i,j, target) in enumerate(test_loader):
        sim,label= test_step(i,j, target)
        sims.extend(sim)
        labels.extend(label)

    fpr, tpr, roc_threshouls = roc_curve(labels,sims)

    roc_auc = auc(fpr,tpr)

    ap = average_precision_score(labels,sims)
      
    logger.info("auc:")
    logger.info(roc_auc)

    logger.info("AP:")
    logger.info(ap)

    logger.info("labels:")
    logger.info(labels)
    logger.info("sims:")
    logger.info(sims)                           


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--augmentation", type=int, default=1,
                        help="Generate multiple image with random rotations and translations")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--db_name", type=str, required=True)

    args = parser.parse_args()

    main(**args.__dict__)