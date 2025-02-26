import argparse
import os

import torch
import yaml
from tqdm import tqdm

from datasets.custom_dataset import get_dataloader, get_dataset
from datasets.custom_transforms import get_transformations
from datasets.utils.configs import INPUT_SIZE
from evaluation.evaluate_utils import PerformanceMeter, predict
from models.build_models import build_model
from utils import create_pred_dir, get_output, to_cuda


def eval_metric(tasks, dataname, test_dl, model, evaluate, save, pred_dir):
    if evaluate:
        performance_meter = PerformanceMeter(dataname, tasks)

    if save:
        # Save all tasks
        tasks_to_save = tasks
    else:
        # Save only edge
        tasks_to_save = ["edge"] if "edge" in tasks else []

    assert evaluate or len(tasks_to_save) > 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dl, desc="Evaluating"):
            batch = to_cuda(batch)
            images = batch["image"]

            if model.backbone.__class__.__name__ == "SAK":
                _, outputs = model(images)
            else:
                outputs = model(images)

            if evaluate:
                performance_meter.update({t: get_output(outputs[t], t) for t in tasks}, batch)

            for task in tasks_to_save:
                predict(dataname, batch["meta"], outputs, task, pred_dir)

    if evaluate:
        # Get evaluation results
        eval_results = performance_meter.get_score()

        results_dict = {}
        for t in tasks:
            for key in eval_results[t]:
                results_dict[t + "_" + key] = eval_results[t][key]

        return results_dict


def test(exp, results_dir, evaluate, save, gpu_id):
    print("Evaluate %s" % exp)

    with open(os.path.join(results_dir, exp, "config.yml"), "r") as stream:
        configs = yaml.safe_load(stream)

    torch.cuda.set_device(gpu_id)

    # Get dataset and tasks
    dataname = configs["dataset"]
    task_dict = configs["task_dict"]
    task_list = []
    for task_name in task_dict:
        task_list += [task_name] * task_dict[task_name]

    test_transforms = get_transformations(dataname, INPUT_SIZE[dataname], train=False)
    test_ds = get_dataset(dataname, train=False, tasks=task_list, transform=test_transforms)
    test_dl = get_dataloader(train=False, configs=configs, dataset=test_ds)

    # Setup output folders
    exp_dir, pred_dir = create_pred_dir(results_dir, exp, task_list)

    # Setup model
    if "student" in configs:
        # Get teacher names
        tea_dims = {}
        for tea_name in configs["teachers"]:
            tea_dims[tea_name] = 0  # dummy value

        # Build student
        stu_config = configs["student"]
        stu_config["backbone"]["tea_dims"] = tea_dims
        stu_config["backbone"]["aligner"] = False  # remove aligners
    else:
        stu_config = configs

    model = build_model(
        arch="mt",
        img_size=INPUT_SIZE[dataname],
        backbone_args=stu_config["backbone"],
        dataname=dataname,
        tasks=task_list,
    ).cuda()

    # load model from checkpoint
    checkpoint_file = os.path.join(exp_dir, "checkpoint.pth")
    if not os.path.exists(checkpoint_file):
        raise ValueError("Checkpoint %s not found!" % (checkpoint_file))

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if "model" in checkpoint.keys():
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Remove aligners in student model
    if "student" in configs:
        out_dict = {}
        for k, v in state_dict.items():
            if "ts_aligner" in k:
                continue
            out_dict[k] = v
        state_dict = out_dict

    model.load_state_dict(state_dict)

    res = eval_metric(task_list, dataname, test_dl, model, evaluate, save, pred_dir)
    # Print and log results
    if evaluate:
        test_results = {key: "%.5f" % res[key] for key in res}
        print(test_results)
        results_file = os.path.join(results_dir, exp, "test_results.txt")
        with open(results_file, "w") as f:
            f.write(str(test_results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", required=True, help="experiment name")
    parser.add_argument("--results_dir", type=str, default="results", help="directory of results")
    parser.add_argument("--evaluate", action="store_true", help="evaluate models")
    parser.add_argument("--save", action="store_true", help="save predictions")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")

    args = parser.parse_args()

    for exp in args.exp:
        test(exp, args.results_dir, args.evaluate, args.save, args.gpu_id)
