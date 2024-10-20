from torch.utils.data import DataLoader
from files.dataset import ImgTextDataset, Transforms


def get_loaders(cfg):

    transforms = Transforms(cfg)
    train_transform = (
        transforms.train_transforms_heavy
        if cfg.trainparams.train_transforms == "heavy"
        else transforms.train_transforms
    )
    valid_transform = transforms.valid_transforms

    train_dataset = ImgTextDataset(cfg, train_transform, "train")
    valid_dataset = ImgTextDataset(cfg, valid_transform, "valid")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=1,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        shuffle=False,
        prefetch_factor=1,
    )

    return train_loader, valid_loader
