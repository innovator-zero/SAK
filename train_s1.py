import argparse
import datetime
import os
import shutil

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from datasets.custom_dataset import get_dataloader, get_dataset
from datasets.custom_transforms import get_transformations
from datasets.utils.configs import INPUT_SIZE
from models.mt_distiller import MTMT_Distiller
from train_utils import cal_params, update_weights
from utils import RunningMeter, create_results_dir, get_loss_metric, global_print, set_seed, to_cuda


def train_one_iter_distiller(
    batch,
    model,
    optimizer,
    train_kd_loss,
    scaler,
    grad_clip,
    fp16,
):
    optimizer.zero_grad()
    batch = to_cuda(batch)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
        kd_loss_dict = model(batch)

    # Log loss values
    batch_size = batch[0].size(0)

    for key in kd_loss_dict.keys():
        if key != "total":
            loss_value = kd_loss_dict[key].detach().item()
            train_kd_loss[key].update(loss_value, batch_size)

    scaler.scale(kd_loss_dict["total"]).backward()

    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

    scaler.step(optimizer)
    scaler.update()


def train_one_epoch(
    epoch,
    iter_count,
    train_dl,
    model,
    optimizer,
    scheduler,
    train_kd_loss,
    scaler,
    grad_clip,
    fp16,
):
    train_dl.sampler.set_epoch(epoch)

    with tqdm(total=len(train_dl), disable=(int(os.environ["RANK"]) != 0)) as t:
        for batch in train_dl:
            t.set_description("Epoch: %d " % (epoch))
            t.update(1)

            train_one_iter_distiller(
                batch,
                model,
                optimizer,
                train_kd_loss,
                scaler,
                grad_clip,
                fp16,
            )

            scheduler.step(iter_count)
            iter_count += 1

    return iter_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Config file path")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_name", type=str, help="Wandb project name")
    parser.add_argument("--fp16", action="store_true", help="Whether to use fp16")
    parser.add_argument("--checkpoint", default=None, help="Load checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    args = parser.parse_args()

    with open(args.config_path, "r") as stream:
        configs = yaml.safe_load(stream)

    # Join args and configs
    configs = {**configs, **vars(args)}

    # Set seed and ddp
    set_seed(args.seed)
    dist.init_process_group("nccl", timeout=datetime.timedelta(0, 3600 * 2))
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    cudnn.benchmark = True
    cv2.setNumThreads(0)

    # Setup logger and output folders
    if global_rank == 0:
        os.makedirs(configs["results_dir"], exist_ok=True)
        configs["exp_dir"] = create_results_dir(configs["results_dir"], args.exp)
        shutil.copy(args.config_path, os.path.join(configs["exp_dir"], "config.yml"))
        if args.wandb_name is not None:
            import wandb

            wandb.init(project=args.wandb_name, id=args.exp, name=args.exp, config=configs)
    dist.barrier()

    # Setup dataset and dataloader
    dataname = configs["dataset"]
    task_list = []

    train_transforms = get_transformations(dataname, INPUT_SIZE[dataname], train=True)
    train_ds = get_dataset(dataname, train=True, tasks=task_list, transform=train_transforms)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, drop_last=True)
    train_dl = get_dataloader(train=True, configs=configs, dataset=train_ds, sampler=train_sampler)

    # Setup model
    model = MTMT_Distiller(
        img_size=INPUT_SIZE[dataname],
        tea_configs=configs["teachers"],
        stu_config=configs["student"],
        loss_type=configs["loss_type"],
        task_out=False,
    ).cuda()

    if global_rank == 0:
        cal_params(model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model = DDP(model, device_ids=[local_rank])

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(configs["lr"]),
        weight_decay=float(configs["weight_decay"]),
    )
    max_epochs = int(configs["max_epochs"])
    warmup_epochs = int(configs["warmup_epochs"])
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=(max_epochs - warmup_epochs) * len(train_dl),
        lr_min=1e-5,
        warmup_t=warmup_epochs * len(train_dl),
        warmup_lr_init=1.25e-7,
        warmup_prefix=True,
    )

    # Setup scaler for amp
    scaler = torch.amp.GradScaler(enabled=args.fp16)

    # Setup loss meters
    train_KD_loss = {}
    for i in range(4):
        for tea_name in configs["teachers"].keys():
            train_KD_loss[tea_name + "_level" + str(i + 1)] = RunningMeter()

    # Determine max epochs and iterations
    max_epochs = configs["max_epochs"]
    start_epoch = 0
    iter_count = 0

    if args.checkpoint is not None:
        # Resume training
        global_print("Loading checkpoint from %s" % args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Update student weights
        update_weights(model.module.student, state_dict)

        if args.resume:
            if "optimizer" in checkpoint.keys():
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint.keys():
                scheduler.load_state_dict(checkpoint["scheduler"])
            if "epoch" in checkpoint.keys():
                start_epoch = checkpoint["epoch"] + 1
            if "iter_count" in checkpoint.keys():
                iter_count = checkpoint["iter_count"]

    global_print("Start: Epoch %d, Iter %d, Goal: Epoch %d" % (start_epoch, iter_count, max_epochs))

    for epoch in range(start_epoch, max_epochs):
        iter_count = train_one_epoch(
            epoch,
            iter_count,
            train_dl,
            model,
            optimizer,
            scheduler,
            train_KD_loss,
            scaler,
            configs["grad_clip"],
            args.fp16,
        )

        logs = get_loss_metric(train_KD_loss, train_KD_loss.keys(), "train_KD")

        # Save checkpoint
        if global_rank == 0:
            # Save checkpoint
            save_ckpt_temp = {
                "model": model.module.student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "iter_count": iter_count,
            }
            torch.save(
                save_ckpt_temp,
                os.path.join(configs["exp_dir"], str(epoch) + "_checkpoint.pth"),
            )
            global_print("Checkpoint saved.")

            if args.wandb_name is not None:
                wandb.log(logs)

        dist.barrier()

    global_print("Training finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
