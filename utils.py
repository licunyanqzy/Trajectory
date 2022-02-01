import logging
import torch
import shutil
import os
import numpy as np


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    if is_best:
        torch.save(state, filename)
        logging.info("-------------- lower ade ----------------")
        shutil.copyfile(filename, "model_best.pth.tar")


def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    # _dir = _dir.split("/")[:-1]
    # _dir = "/".join(_dir)
    return os.path.join(_dir, "datasets", dset_name, dset_type)


def average_displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode="sum"):
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss ** 2

    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)

    if mode == "sum":
        return torch.sum(loss)
    elif mode == "mean":
        return torch.mean(loss)
    elif mode == "raw":
        return loss


def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, mode="sum"):
    loss = pred_pos_gt - pred_pos
    loss = loss ** 2

    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))

    if mode == "raw":
        return loss
    else:
        return torch.sum(loss)


def cal_ADE_FDE(pred_traj_gt, pred_traj, consider_ped=None, mode="sum"):
    ADE = average_displacement_error(pred_traj, pred_traj_gt, consider_ped, mode)
    FDE = final_displacement_error(pred_traj[-1], pred_traj_gt[-1], consider_ped, mode)
    return ADE, FDE


def loss_test(pred_fake, pred_gt):
    loss = pred_gt - pred_fake
    loss = loss ** 2
    loss = torch.mean(torch.sqrt(torch.sum(loss, dim=2)), dim=1)
    ave = torch.mean(loss)
    return torch.cat((loss, ave.unsqueeze(0)))


def l2_loss(pred_traj, pred_traj_gt, loss_mask, mode="average"):
    seq_len, batch, _ = pred_traj.size()
    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2

    exp = torch.arange(12).float().cuda()
    exp = torch.exp(exp / 6).unsqueeze(0).unsqueeze(-1)
    loss = loss * exp.repeat(batch, 1, 2)

    if mode == "sum":
        return torch.sum(loss)
    elif mode == "average":
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == "raw":
        return loss.sum(dim=2).sum(dim=1)


def l2_loss_sum(l2_loss, seq_start_end, seq_len):
    loss_sum = torch.zeros(1).cuda()
    num_sum = torch.zeros(1).cuda()
    for start, end in seq_start_end.data:
        _l2_loss = torch.narrow(l2_loss, 0, start, end - start)
        _l2_loss = torch.sum(_l2_loss, dim=0)
        _l2_loss = torch.min(_l2_loss)
        loss_sum += _l2_loss
        num_sum += seq_len * (end - start)
    return loss_sum / num_sum


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def int_tuple(s):
    return tuple(int(i) for i in s.split(","))


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def cal_goal(traj):
    seq_len, num, c = traj.size()

    action = torch.zeros(seq_len-1, num, c).cuda()
    for i in range(seq_len - 1):
        action[i, :, :] = traj[i + 1, :, :] - traj[i, :, :]

    goal = torch.zeros(seq_len, num, c).cuda()

    for j in range(num):
        index = 0
        for i in range(seq_len - 2):
            velocity1 = torch.norm(action[i, j, :])
            velocity2 = torch.norm(action[i+1, j, :])

            turn = torch.dot(action[i, j, :], action[i+1, j, :]) / (velocity1 * velocity2)

            if turn < 0.1:
                goal[index:i+2, j, :] = traj[i+1, j, :]
                index = i + 1
                continue

            if velocity1 < 0.004:
                goal[index:i+1, j, :] = traj[i+1, j, :]
                index = i

        goal[index:seq_len, j, :] = traj[-1, j, :]

    return goal

