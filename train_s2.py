import argparse
import datetime
import os
import shutil

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from datasets.custom_dataset import get_dataloader, get_dataset
from datasets.custom_transforms import get_transformations
from datasets.utils.configs import INPUT_SIZE, NUM_TRAIN_IMAGES
from evaluation.evaluate_utils import PerformanceMeter
from losses import get_criterion
from models.mt_distiller import MTMT_Distiller
from train_utils import cal_params, get_optimizer_scheduler, update_weights
from utils import RunningMeter, create_results_dir, get_loss_metric, get_output, global_print, set_seed, to_cuda


def train_one_iter_distiller(
    task_out,
    tasks,
    alpha,
    batch,
    model,
    optimizer,
    train_loss,
    train_kd_loss,
    scaler,
    grad_clip,
    fp16,
):
    optimizer.zero_grad()
    batch = to_cuda(batch)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
        if task_out:
            kd_loss_dict, task_loss_dict = model(batch)
        else:
            kd_loss_dict = model(batch)

    # Log loss values
    batch_size = batch["image"].size(0)

    for task in tasks:
        loss_value = task_loss_dict[task].detach().item()
        train_loss[task].update(loss_value, batch_size)

    for key in kd_loss_dict.keys():
        if key != "total":
            loss_value = kd_loss_dict[key].detach().item()
            train_kd_loss[key].update(loss_value, batch_size)

    if task_out:
        scaler.scale(task_loss_dict["total"] + alpha * kd_loss_dict["total"]).backward()
    else:
        scaler.scale(kd_loss_dict["total"]).backward()

    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

    scaler.step(optimizer)
    scaler.update()


def train_one_epoch(
    epoch,
    iter_count,
    max_iter,
    task_out,
    tasks,
    alpha,
    train_dl,
    model,
    optimizer,
    scheduler,
    train_loss,
    train_kd_loss,
    scaler,
    grad_clip,
    fp16,
):
    train_dl.sampler.set_epoch(epoch)

    with tqdm(total=len(train_dl), disable=(int(os.environ["RANK"]) != 0)) as t:
        for batch in train_dl:
            t.set_description("Epoch: %d Iter: %d" % (epoch, iter_count))
            t.update(1)

            train_one_iter_distiller(
                task_out,
                tasks,
                alpha,
                batch,
                model,
                optimizer,
                train_loss,
                train_kd_loss,
                scaler,
                grad_clip,
                fp16,
            )

            if scheduler.__class__.__name__ == "PolynomialLR":
                scheduler.step()

            iter_count += 1

            if iter_count >= max_iter:
                end_signal = True
                break
            else:
                end_signal = False

    if scheduler.__class__.__name__ == "CosineLRScheduler":
        scheduler.step(epoch)

    return end_signal, iter_count


def eval_metric(task_out, tasks, dataname, val_dl, model, val_kd_loss):
    """
    Evaluate the model
    """

    performance_meter = PerformanceMeter(dataname, tasks)

    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Evaluating"):
            batch = to_cuda(batch)
            if task_out:
                kd_loss_dict, outputs = model.module.forward_val(batch)

                performance_meter.update({t: get_output(outputs[t], t) for t in tasks}, batch)
            else:
                kd_loss_dict = model.module.forward_val(batch)

            # Log loss values
            if isinstance(batch, dict):
                batch_size = batch["image"].size(0)
            else:
                batch_size = batch[0].size(0)

            for key in kd_loss_dict.keys():
                if key != "total":
                    loss_value = kd_loss_dict[key].detach().item()
                    val_kd_loss[key].update(loss_value, batch_size)

    results_dict = {}
    if task_out:
        eval_results = performance_meter.get_score()
        for task in tasks:
            for key in eval_results[task]:
                results_dict["eval/" + task + "_" + key] = eval_results[task][key]

    return results_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Config file path")
    parser.add_argument("--exp", type=str, required=True, help="Experiment name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_name", type=str, help="Wandb project name")
    parser.add_argument("--fp16", action="store_true", help="Whether to use fp16")
    parser.add_argument("--checkpoint", default=None, help="Load checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--task_out", action="store_true", help="Whether to output task or distill only")
    parser.add_argument("--alpha", type=float, default=1.0, help="Balance between task loss and distillation loss")

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
    task_dict = configs["task_dict"]
    task_list = []
    if args.task_out:
        for task_name in task_dict:
            task_list += [task_name] * task_dict[task_name]

    train_transforms = get_transformations(dataname, INPUT_SIZE[dataname], train=True)
    val_transforms = get_transformations(dataname, INPUT_SIZE[dataname], train=False)

    train_ds = get_dataset(dataname, train=True, tasks=task_list, transform=train_transforms)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, drop_last=True)
    train_dl = get_dataloader(train=True, configs=configs, dataset=train_ds, sampler=train_sampler)

    val_ds = get_dataset(dataname, train=False, tasks=task_list, transform=val_transforms)
    val_dl = get_dataloader(train=False, configs=configs, dataset=val_ds)

    # Setup loss function
    if args.task_out:
        criterion = get_criterion(dataname, task_list).cuda()
    else:
        criterion = None

    # Setup model
    model = MTMT_Distiller(
        img_size=INPUT_SIZE[dataname],
        tea_configs=configs["teachers"],
        stu_config=configs["student"],
        loss_type=configs["loss_type"],
        task_out=args.task_out,
        stu_criterion=criterion,
        dataname=dataname,
        tasks=task_list,
    ).cuda()

    if global_rank == 0:
        cal_params(model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model = DDP(model, device_ids=[local_rank])

    # Setup optimizer and scheduler
    optimizer, scheduler = get_optimizer_scheduler(configs, model)

    # Setup scaler for amp
    scaler = torch.amp.GradScaler(enabled=args.fp16)

    # Setup loss meters
    train_loss = {}
    train_KD_loss = {}
    val_KD_loss = {}
    for task in task_list:
        train_loss[task] = RunningMeter()
    for i in range(4):
        for tea_name in configs["teachers"].keys():
            train_KD_loss[tea_name + "_level" + str(i + 1)] = RunningMeter()
            val_KD_loss[tea_name + "_level" + str(i + 1)] = RunningMeter()

    # Determine max epochs and iterations
    max_epochs = configs["max_epochs"]
    max_iter = configs["max_iters"]

    if max_epochs > 0:
        max_iter = 10000000
    else:
        assert max_iter > 0
        max_epochs = 1000000

    start_epoch = 0
    iter_count = 0

    if args.checkpoint is not None:
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

    global_print(
        "Start: Epoch %d, Iter %d, Goal: Epoch %d or Iter %d" % (start_epoch, iter_count, max_epochs, max_iter)
    )

    for epoch in range(start_epoch, max_epochs):
        end_signal, iter_count = train_one_epoch(
            epoch,
            iter_count,
            max_iter,
            args.task_out,
            task_list,
            args.alpha,
            train_dl,
            model,
            optimizer,
            scheduler,
            train_loss,
            train_KD_loss,
            scaler,
            configs["grad_clip"],
            args.fp16,
        )

        if args.task_out:
            logs = get_loss_metric(train_loss, task_list, "train")
        else:
            logs = {}

        train_KD_stats = get_loss_metric(train_KD_loss, train_KD_loss.keys(), "train_KD")
        logs.update(train_KD_stats)

        # Validation
        if global_rank == 0:
            if (epoch + 1) % configs["eval_freq"] == 0 or epoch == max_epochs - 1 or end_signal:
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
                    os.path.join(configs["exp_dir"], "checkpoint.pth"),
                )
                global_print("Checkpoint saved.")

                global_print("Validation at epoch %d." % epoch)
                val_logs = eval_metric(args.task_out, task_list, dataname, val_dl, model, val_KD_loss)
                val_KD_stats = get_loss_metric(val_KD_loss, val_KD_loss.keys(), "eval_KD")
                val_logs.update(val_KD_stats)
                global_print(val_logs)

                if args.wandb_name is not None:
                    wandb.log({**logs, **val_logs})
            else:
                if args.wandb_name is not None:
                    wandb.log(logs)

        dist.barrier()

        if end_signal:
            break

    global_print("Training finished.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
