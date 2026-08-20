import argparse
import copy
import logging
import os
import sys
from collections import OrderedDict
from importlib import import_module

import segmentation_models_pytorch as smp
import torch
import torch.distributed as dist
from segment_anything import sam_model_registry

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from augmodule_utils.utils import AverageMeter, mae, select
import train_regcl as train_regcl_lib
from train_regcl import get_dataloader_SAM, load_dataset, setup_seed
from merging.task_vectors import TaskVector, merge_max_abs, merge_rnd_mix
from merging.ties import merge_methods, state_dict_to_vector, vector_to_state_dict

TIES_CONFIG = ("topk", 20, "mass", "dis-mean")

def setup_distribution(args: argparse.Namespace) -> None:
    if args.cuda == -1 and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        logging.info(
            f"Initializing rank {args.rank} in "
            f"{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        )
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=args.rank,
        )
        dist.barrier()
    else:
        if args.cuda == -1:
            args.cuda = 0
        torch.cuda.set_device(args.cuda)
        args.rank = 0
        args.world_size = 1
        logging.info(f"Running in single GPU mode on cuda:{args.cuda}")


def build_base_path(args: argparse.Namespace) -> str:
    base_dir = f"log/Comparison/{args.module}/magmax/{args.order}"
    return f"{base_dir}/{args.seed}_SEED"


def checkpoint_name(task_name: str, meth: str) -> str:
    if meth == "sequ":
        return f"ft_{task_name}.pth"
    if meth == "indi":
        return f"{task_name}.pth"
    raise ValueError(f"Unsupported training method: {meth}")


def torch_load_for_merge(path):
    # Checkpoints are merged on rank0 without forward/backward, so CPU avoids
    # unnecessary GPU memory pressure while preserving the saved LoRA format.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_tensors(checkpoint):
    tensors = OrderedDict()
    for key, value in sorted(checkpoint.items()):
        if isinstance(value, torch.nn.Module):
            for sub_key, tensor in sorted(value.state_dict().items()):
                if torch.is_tensor(tensor) and torch.is_floating_point(tensor):
                    tensors[f"{key}.{sub_key}"] = tensor.detach().clone().cpu()
        elif torch.is_tensor(value) and torch.is_floating_point(value):
            tensors[key] = value.detach().clone().cpu()
    return tensors


def checkpoint_from_tensors(reference_checkpoint, tensors):
    checkpoint = copy.deepcopy(reference_checkpoint)
    module_tensors = {}
    for key, tensor in tensors.items():
        name, dot, sub_key = key.partition(".")
        if dot and isinstance(checkpoint.get(name), torch.nn.Module):
            module_tensors.setdefault(name, {})[sub_key] = tensor.detach().clone()
        else:
            checkpoint[name] = tensor.detach().clone()

    for name, updates in module_tensors.items():
        state_dict = checkpoint[name].state_dict()
        state_dict.update(updates)
        checkpoint[name].load_state_dict(state_dict, strict=True)
    return checkpoint


def build_merge_logger(base_path, merge_name):
    log_dir = os.path.join(base_path, merge_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"magmax_merging.{merge_name}.{base_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(os.path.join(log_dir, "log.txt"), mode="w")
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    return logger


def get_merge_names(method):
    if method == "all":
        return ["mag_rnd_mix", "mag_max_abs", "mag_sum", "ties"]
    if method == "magmax":
        return ["mag_rnd_mix", "mag_max_abs", "mag_sum"]
    return [method]


def ties_merging(task_vectors):
    reset_type, reset_thresh, resolve, merge = TIES_CONFIG
    flat_task_vectors = torch.vstack(
        [state_dict_to_vector(task_vector.vector) for task_vector in task_vectors]
    )
    merged_flat = merge_methods(
        reset_type,
        flat_task_vectors,
        reset_thresh=reset_thresh,
        resolve_method=resolve,
        merge_func=merge,
    )
    merged_vector = vector_to_state_dict(
        merged_flat, task_vectors[0].vector, remove_keys=[]
    )
    return TaskVector(vector=OrderedDict(merged_vector))


def merge_task_vectors(task_vectors, method):
    if method == "mag_rnd_mix":
        return merge_rnd_mix(task_vectors), 1.0
    if method == "mag_max_abs":
        return merge_max_abs(task_vectors), 0.5
    if method == "mag_sum":
        return sum(task_vectors), 1.0 / len(task_vectors)
    if method == "ties":
        return ties_merging(task_vectors), 0.55
    raise ValueError(f"Unsupported merging method: {method}")


def merged_path(base_path, task_names, method):
    filename = f"init_{'_'.join(task_names)}.pth"
    return os.path.join(base_path, method, filename)


def save_incremental_merge(init_path, ft_paths, task_names, base_path, method, merge_logger=None):
    init_checkpoint = torch_load_for_merge(init_path)
    init_tensors = checkpoint_tensors(init_checkpoint)
    task_vectors = []

    for ft_path in ft_paths:
        ft_tensors = checkpoint_tensors(torch_load_for_merge(ft_path))
        if set(ft_tensors) != set(init_tensors):
            raise ValueError(f"LoRA checkpoint keys mismatch when loading {ft_path}")
        task_vectors.append(TaskVector(vector=OrderedDict(
            (key, ft_tensors[key] - init_tensors[key]) for key in init_tensors
        )))

    save_path = merged_path(base_path, task_names, method)
    merged_task_vector, alpha = merge_task_vectors(task_vectors, method)
    merged_tensors = OrderedDict(
        (key, init_tensors[key] + alpha * merged_task_vector.vector[key])
        for key in init_tensors
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint_from_tensors(init_checkpoint, merged_tensors), save_path)
    if merge_logger is not None:
        merge_logger.info(f"Saved checkpoint: {save_path}, alpha={alpha}")
    return save_path


def collect_base_model_paths(base_path, task_names, meth):
    init_path = os.path.join(base_path, "init.pth")
    ft_paths = [os.path.join(base_path, checkpoint_name(task_name, meth)) for task_name in task_names]
    missing = [path for path in [init_path] + ft_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing base checkpoint(s):\n" + "\n".join(missing)
        )
    return init_path, ft_paths

def validate_checkpoint(args, sam, net_origin, merge_path, testloader, task_id, id_to_key, logger):
    val_net = select(args, net_origin, merge_path)
    val_net.eval()
    results = {}

    for index in range(len(testloader)):
        ious = AverageMeter()
        f1_scores = AverageMeter()
        mae_scores = AverageMeter()
        if args.cuda != -1 or args.rank == 0:
            logger.info(f"-----{index} of {id_to_key[index]} begin test-------")

        last_iter = -1
        for iter_id, data in enumerate(testloader[index]):
            last_iter = iter_id
            images = data["image"].cuda(non_blocking=True)
            gt_masks = data["label"].cuda(non_blocking=True)
            points = data["point"]
            data["point"][0] = data["point"][0].cuda(non_blocking=True)
            data["point"][1] = data["point"][1].cuda(non_blocking=True)

            with torch.no_grad():
                if args.begin > 0:
                    end_before_lora = args.begin - 1
                    input_images = sam.preprocess(images)
                    mid_embed, _ = sam.image_encoder(
                        input_images, False, begin=-1, end=end_before_lora
                    )
                    outputs = val_net(mid_embed, points=points, begin=args.begin, end=-1)
                else:
                    outputs = val_net(images, points=points)

            for pred_mask, gt_mask in zip(outputs["masks"], gt_masks):
                if len(gt_mask.size()) < 3:
                    gt_mask = gt_mask.unsqueeze(0)
                batch_stats = smp.metrics.get_stats(
                    torch.sigmoid(pred_mask),
                    gt_mask.int(),
                    mode="binary",
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(
                    *batch_stats, reduction="micro-imagewise"
                )
                batch_f1 = smp.metrics.f1_score(
                    *batch_stats, reduction="micro-imagewise"
                )
                batch_mae = mae(pred_mask, gt_mask)
                if args.cuda == -1:
                    iou_list = [
                        torch.zeros_like(batch_iou.clone().detach())
                        for _ in range(dist.get_world_size())
                    ]
                    f1_list = [
                        torch.zeros_like(batch_f1.clone().detach())
                        for _ in range(dist.get_world_size())
                    ]
                    mae_list = [
                        torch.zeros_like(batch_mae.clone().detach())
                        for _ in range(dist.get_world_size())
                    ]
                    dist.all_gather(iou_list, batch_iou)
                    dist.all_gather(f1_list, batch_f1)
                    dist.all_gather(mae_list, batch_mae)
                    for gathered_id in range(len(iou_list)):
                        mae_scores.update(mae_list[gathered_id], 1)
                        ious.update(iou_list[gathered_id], 1)
                        f1_scores.update(f1_list[gathered_id], 1)
                else:
                    mae_scores.update(batch_mae, 1)
                    ious.update(batch_iou, 1)
                    f1_scores.update(batch_f1, 1)

            if iter_id % 50 == 0 and (args.cuda != -1 or args.rank == 0):
                logger.info(
                    f"Val: [[{iter_id}/{len(testloader[index])}]: "
                    f"Mean IoU: [{ious.avg:.4f}] -- "
                    f"Mean F1: [{f1_scores.avg:.4f}] -- "
                    f"MAE: [{mae_scores.avg:.4f}]"
                )

        if args.rank == 0 or args.cuda != -1:
            logger.info(
                f"Val: [[{last_iter}/{len(testloader[index])}]: "
                f"Mean IoU: [{ious.avg:.4f}] -- "
                f"Mean F1: [{f1_scores.avg:.4f}] -- "
                f"MAE: [{mae_scores.avg:.4f}]"
            )
        if index <= task_id + 1:
            results[index] = (ious.avg, f1_scores.avg, mae_scores.avg)
    return results


def run_merging(args, train_data, test_data, merge_method, base_path):
    train_regcl_lib.args = args
    task_names = list(train_data.keys())
    id_to_key = {index: key for index, key in enumerate(task_names)}
    init_path, ft_paths = collect_base_model_paths(base_path, task_names, args.meth)

    lora_layer = list(range(args.begin, args.end + 1))
    sam, img_embedding_size = sam_model_registry[args.vit_name](checkpoint=args.ckpt)
    sam = sam.cuda()
    for _, param in sam.image_encoder.named_parameters():
        param.requires_grad = False
    sam.image_encoder.train(mode=False)

    pkg = import_module(f"module.{args.module}")
    logging.info(f"Module {args.module} loaded to layers: {lora_layer}")
    net_origin = pkg.Adapter_Sam(copy.deepcopy(sam), lora_layer=lora_layer)

    testloader = {}
    for task_id, key in enumerate(task_names):
        train_data_location = train_data[key]
        test_data_location = test_data[key]
        if args.cuda == -1:
            _, testloader[task_id], _, _ = get_dataloader_SAM(
                args, img_embedding_size, train_data_location, test_data_location
            )
        else:
            _, testloader[task_id] = get_dataloader_SAM(
                args, img_embedding_size, train_data_location, test_data_location
            )

    merge_logger = logging
    if args.rank == 0:
        merge_logger = build_merge_logger(base_path, merge_method)
        merge_logger.info(str(args))
        merge_logger.info(f"base_path: {base_path}")

    AIJ = {}
    for task_id in range(len(task_names)):
        current_task_names = task_names[: task_id + 1]
        current_ft_paths = ft_paths[: task_id + 1]
        merge_path = merged_path(base_path, current_task_names, merge_method)
        if args.rank == 0:
            save_incremental_merge(
                init_path,
                current_ft_paths,
                current_task_names,
                base_path,
                merge_method,
                merge_logger,
            )
        if args.cuda == -1:
            dist.barrier()

        if args.cuda != -1 or args.rank == 0:
            merge_logger.info(f"---Validation on {merge_path}---")
        results = validate_checkpoint(
            args,
            sam,
            net_origin,
            merge_path,
            testloader,
            task_id,
            id_to_key,
            merge_logger,
        )
        for index, scores in results.items():
            AIJ[(task_id, index)] = scores
        if args.rank == 0 or args.cuda != -1:
            train_regcl_lib.get_forgetting_metric(task_id, AIJ, merge_logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default=None, help="Path containing init.pth and base model pths")
    parser.add_argument("--module", type=str, default="AugModule", choices=["AugModule", "LoRA"], help="Module (Default=AugModule)")
    parser.add_argument("--batch_size", type=int, default=2, help="batch_size per gpu (Default=2)")
    parser.add_argument("--layers", type=str, default="7-11", help="LoRA layers (Default=7-11)")
    parser.add_argument("--vit_name", type=str, default="vit_b", help="select one vit model (Default=vit_b)")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoint/sam_vit_b_01ec64.pth",
        help="Pretrained checkpoint",
    )
    parser.add_argument("--img_size", type=int, default=1024, help="input patch size of network input (Default=1024)")
    parser.add_argument("--seed", type=int, default=1234, help="random seed (Default=1234)")
    parser.add_argument("--order", type=str, default="Kvasir_camo_ISTD_ISIC_cod", help="Training order")
    parser.add_argument("--cuda", type=int, default=-1, help="ID of GPU when using single GPU")
    parser.add_argument(
        "--meth",
        type=str,
        default="sequ",
        choices=["indi", "sequ"],
        help="Base checkpoint naming mode: indi uses task.pth; sequ uses ft_task.pth.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", "magmax", "mag_rnd_mix", "mag_max_abs", "mag_sum", "ties"],
        help="Merging method to save and evaluate.",
    )
    args = parser.parse_args()
    args.alpha = 1
    args.beta = 10
    args.deta = 1
    args.begin = int(args.layers.split("-")[0])
    args.end = int(args.layers.split("-")[1])

    setup_seed(args.seed)

    base_path = args.base_path or build_base_path(args)
    os.makedirs(base_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    setup_distribution(args)
    logging.info(str(args))
    logging.info(f"base_path: {base_path}")
    train_data, test_data = load_dataset(args)
    for merge_method in get_merge_names(args.method):
        run_merging(args, train_data, test_data, merge_method, base_path)
