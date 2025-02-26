from torch.utils.data import DataLoader

from .utils.custom_collate import collate_mil
from utils import global_print


def get_dataset(dataname, train, tasks, transform, dataidxs=None):
    """
    Get the dataset
    """
    if train:
        global_print("Get training dataset for %s" % (dataname))
    else:
        global_print("Get validation dataset for %s" % (dataname))

    if dataname == "pascalcontext":
        from .pascal_context import PASCALContext

        database = PASCALContext(
            train=train, transform=transform, tasks=tasks, dataidxs=dataidxs
        )
    elif dataname == "nyud":
        from .nyud import NYUD

        database = NYUD(
            train=train, transform=transform, tasks=tasks, dataidxs=dataidxs
        )
    elif dataname == "imagenet":
        from .imagenet import ImageTarDataset

        database = ImageTarDataset(train=train, transform=transform)
    # New datasets can be added here
    else:
        raise NotImplementedError(
            "'dataname': Choose among 'pascalcontext' and 'nyud'!"
        )

    return database


def get_dataloader(train, configs, dataset, sampler=None):
    """
    Get the dataloader from dataset
    """
    if train:
        dataloader = DataLoader(
            dataset,
            batch_size=configs["tr_batch"],
            drop_last=True,
            num_workers=configs["nworkers"],
            collate_fn=collate_mil,
            pin_memory=True,
            sampler=sampler,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=configs["val_batch"],
            shuffle=False,
            drop_last=False,
            num_workers=configs["nworkers"],
            collate_fn=collate_mil,
            pin_memory=True,
        )
    return dataloader
