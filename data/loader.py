from torch.utils.data import DataLoader
from data.trajectories import TrajectoryDataset, seq_collate


def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,    # 多线程读取数据
        collate_fn=seq_collate,     # 将一个list的sample组成一个mini-batch的函数
        pin_memory=True)
    return dset, loader
