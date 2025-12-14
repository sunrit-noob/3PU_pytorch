import argparse
import os
import time
from math import log
import numpy as np
from tqdm import tqdm
from glob import glob
from collections import defaultdict

import torch
import torch.utils.data as data

from network.upsampler import Net
from model import Model
from network.model_loss import ChamferLoss
from network import operations
from utils import pc_utils, pytorch_utils
from misc import logger
# from data import H5Dataset
from dataset import H5Dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='test',
                    help='train or test [default: train]')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use [default: GPU 0]')
parser.add_argument('--id', default='demo',
                    help="experiment name, prepended to log_dir")
parser.add_argument('--log_dir', default='./model',
                    help='Log dir [default: log]')
parser.add_argument('--model', default='model_microscope', help='model name')
parser.add_argument('--root_dir', default='../',
                    help='project root, data and h5_data diretories')
parser.add_argument('--result_dir', help='result directory')
parser.add_argument('--ckpt', help='model to restore from')
parser.add_argument('--num_point', type=int,
                    help='Point Number [1024/2048] [default: 1024]')
parser.add_argument('--num_shape_point', type=int,
                    help="Number of points per shape")
parser.add_argument('--up_ratio', type=int, default=16,
                    help='Upsampling Ratio [default: 2]')
parser.add_argument('--max_epoch', type=int, default=160,
                    help='Epoch to run [default: 500]')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch Size during training')
parser.add_argument('--h5_data', help='h5 file for training')
parser.add_argument('--record_data', help='record file for training')
parser.add_argument('--test_data', help='test data path')
parser.add_argument('--lr_init', type=float, default=0.0005)
parser.add_argument('--restore_epoch', type=int)
parser.add_argument('--stage_steps', type=int, default=15000,
                    help="number of updates per curriculums stage")
parser.add_argument('--step_ratio', type=int, default=2,
                    help="upscale ratio per step")
parser.add_argument('--patch_num_ratio', type=float, default=3)
parser.add_argument('--jitter', action="store_true",
                    help="jitter augmentation")
parser.add_argument('--jitter_sigma', type=float,
                    default=0.0025, help="jitter augmentation")
parser.add_argument('--jitter_max', type=float,
                    default=0.005, help="jitter augmentation")
parser.add_argument('--drop_out', type=float, default=1.0,
                    help="drop_out ratio. default 1.0 (no drop out) ")
parser.add_argument('--knn', type=int, default=32,
                    help="neighbood size for edge conv")
parser.add_argument('--dense_n', type=int, default=3,
                    help="number of dense layers")
parser.add_argument('--block_n', type=int, default=3,
                    help="number of dense blocks")
parser.add_argument('--fm_knn', type=int, default=5,
                    help="number of neighboring points for feature matching")

parser.add_argument('--growth_rate', type=int, default=12,
                    help='dense block growth rate')
parser.add_argument('--cd_threshold', default=2.0,
                    type=float, help="threshold for cd")
parser.add_argument('--fidelity_weight', default=50.0,
                    type=float, help="chamfer loss weight")

FLAGS = parser.parse_args()
PHASE = FLAGS.phase
DEVICE = torch.device('cuda', FLAGS.gpu)
ROOT_DIR = FLAGS.root_dir
MODEL_DIR = os.path.join(FLAGS.log_dir, FLAGS.id)
CKPT = FLAGS.ckpt

NUM_SHAPE_POINT = FLAGS.num_shape_point
NUM_POINT = FLAGS.num_point
assert(NUM_SHAPE_POINT is not None or NUM_POINT is not None)
NUM_POINT = NUM_POINT or int(NUM_SHAPE_POINT * FLAGS.drop_out)

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
LR_INIT = FLAGS.lr_init
JITTER = FLAGS.jitter
JITTER_MAX = FLAGS.jitter_max
JITTER_SIGMA = FLAGS.jitter_sigma
STAGE_STEPS = FLAGS.stage_steps

STEP_RATIO = FLAGS.step_ratio
RESTORE_EPOCH = FLAGS.restore_epoch
FM_KNN = FLAGS.fm_knn
KNN = FLAGS.knn
GROWTH_RATE = FLAGS.growth_rate
DENSE_N = FLAGS.dense_n
CD_THRESHOLD = FLAGS.cd_threshold

UP_RATIO = FLAGS.up_ratio
TRAIN_H5 = FLAGS.h5_data
TRAIN_RECORD = FLAGS.record_data

TEST_DATA = FLAGS.test_data
PATCH_NUM_RATIO = FLAGS.patch_num_ratio


# build model
net = Net(max_up_ratio=UP_RATIO, step_ratio=STEP_RATIO,
          knn=KNN, growth_rate=GROWTH_RATE, dense_n=DENSE_N, fm_knn=FM_KNN)


def get_stage_progress(step):
    """
    return the stage (an integer from 0) and progress (float 0~1)
    """
    stage = (step + STAGE_STEPS) // (2 * STAGE_STEPS)
    progress = (step + STAGE_STEPS) / (2 * STAGE_STEPS) - stage
    return stage, progress

def train():
    net.to(DEVICE)
    net.train()
    chamfer_criteria = ChamferLoss()
    old_lr = FLAGS.lr_init
    lr = FLAGS.lr_init
    optimizer = torch.optim.Adam(net.parameters(),
                                lr=FLAGS.lr_init,
                                betas=(0.9, 0.999))
    if FLAGS.ckpt is not None:
        step = pytorch_utils.load_network(net, FLAGS.ckpt)
    else:
        step = 0
    # data loader
    if TRAIN_H5 is not None:
        from dataset import H5Dataset
        dataset = H5Dataset(
            h5_path=TRAIN_H5,
            num_shape_point=NUM_SHAPE_POINT, num_patch_point=NUM_POINT,
            batch_size=BATCH_SIZE, up_ratio=UP_RATIO, step_ratio=STEP_RATIO)
        # dataset = PUNET_Dataset(h5_file_path=TRAIN_H5)
        dataloader = data.DataLoader(
            dataset, batch_size=1, pin_memory=True)

    start_epoch = step // len(dataloader)
    # whenever progress is changed, we need to update:
    # 1. chamferloss threshold
    # 2. dataset.combined
    # 3. dataset.curr_threshold
    stage, progress = get_stage_progress(step)
    start_ratio = STEP_RATIO ** (stage + 1)
    dataset.set_max_ratio(start_ratio)
    if progress > 0.5:
        dataset.set_combined()
        if progress > 0.6:
            chamfer_criteria.set_threshold(CD_THRESHOLD)
    else:
        chamfer_criteria.unset_threshold()
        dataset.unset_combined()

    # visualization
    # vis_logger = visdom.Visdom(env=FLAGS.id)
    for epoch in range(start_epoch + 1, MAX_EPOCH):
        
        for examples in tqdm(dataloader):
            input_pc, label_pc, ratio = examples
            ratio = ratio.item()
            # 1xBx3xN
            input_pc = input_pc[0].to(DEVICE)
            label_pc = label_pc[0].to(DEVICE)
            # model.set_input(input_pc, ratio, label_pc=label_pc)
            input = input_pc.detach()
            # gt point cloud
            if label_pc is not None:
                gt = label_pc.detach()
            else:
                gt = None
            
            # run gradient decent and increment model.step
            optimizer.zero_grad()

            net.train()
            if gt is not None:
                predicted, gt = net(input, ratio=ratio, gt=gt)
            else:
                predicted = net(input, ratio=ratio)

            loss_chamfer = chamfer_criteria(predicted.transpose(1, 2).contiguous(),
            gt.transpose(1, 2).contiguous())
            weight = log(net.max_up_ratio / ratio, net.step_ratio)
            loss = loss_chamfer * weight

            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), 1)
            optimizer.step()
            step += 1
            new_stage, new_progress = get_stage_progress(step)
            # advance to the next training stage with an added ratio
            if stage + 1 == new_stage:
                # dataset.add_next_ratio()
                # dataset.unset_combined()
                chamfer_criteria.unset_threshold()
            # advance to the combined stage
            # if progress <= 0.5 and new_progress > 0.5:
            #     dataset.set_combined()
            # chamfer loss set ignore threshold
            if new_progress > 0.6:
                chamfer_criteria.set_threshold(CD_THRESHOLD)
            # if model.step % 50 == 0:
            #     output = model.predicted.transpose(2, 1)[0].cpu()
            #     gt = model.gt.transpose(2, 1)[0].cpu()
            #     input_pc = input_pc.transpose(2, 1)[0].cpu()
                # vis_logger.scatter(input_pc, win="x{}_input".format(ratio),
                #                    opts=dict(title="x{}_input".format(ratio),
                #                              markersize=2))
                # vis_logger.scatter(output, win="x{}_output".format(ratio),
                #                    opts=dict(title="x{}_output".format(ratio),
                #                              markersize=2))
                # vis_logger.scatter(gt, win="x{}_gt".format(ratio),
                #                    opts=dict(title="x{}_label".format(ratio),
                #                              markersize=2))
                # vis_logger.line(
                #     np.array([model.error_log["cd_loss_x{}".format(ratio)]]),
                #     np.array([model.step]),
                #     update="append",
                #     win="x{}_loss".format(ratio),
                #     opts=dict(title="x{}_loss".format(ratio)))

            stage, progress = new_stage, new_progress

        # end of epoch
        if epoch % 20 == 0:
            pytorch_utils.save_network(net, MODEL_DIR,
                                       "model", epoch_label=str(epoch),
                                       step=str(step))
        
if __name__ == "__main__":
    
    if PHASE == "train":
        train()
