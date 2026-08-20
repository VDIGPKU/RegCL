import argparse
import copy
import logging
import os
from importlib import import_module
from pathlib import Path

import torch
import torch.distributed as dist
from segment_anything import sam_model_registry

from augmodule_utils.utils import configure_opt, select, train
import train_regcl as train_regcl_lib
from train_regcl import get_dataloader_SAM, load_dataset, setup_seed
from merging.magmax_merging import validate_checkpoint


ROOT = Path(__file__).parent


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


def build_output_path(args: argparse.Namespace) -> str:
    base_dir = f"log/Comparison/{args.module}/magmax/{args.order}"
    return f"{base_dir}/{args.seed}_SEED"


def checkpoint_name(task_name: str, meth: str) -> str:
    if meth == "sequ":
        return f"ft_{task_name}.pth"
    if meth == "indi":
        return f"{task_name}.pth"
    raise ValueError(f"Unsupported training method: {meth}")


def train_base_models(args, train_data, test_data, output_path, logger):
    train_regcl_lib.args = args
    id_to_key = {index: key for index, key in enumerate(train_data)}
    order_lst = list(train_data.keys())
    world_size = args.world_size if args.cuda == -1 else 1
    scaled_lr = args.lr * (world_size / 4.0)

    if args.rank == 0:
        logger.info(f"lr scaling: base_lr={args.lr}, gpu_nums={world_size}, actual_lr={scaled_lr}")

    lora_layer = list(range(args.begin, args.end + 1))
    sam, img_embedding_size = sam_model_registry[args.vit_name](checkpoint=args.ckpt)
    sam = sam.cuda()
    for _, param in sam.image_encoder.named_parameters():
        param.requires_grad = False
    sam.image_encoder.train(mode=False)

    pkg = import_module(f"module.{args.module}")
    logger.info(f"Module {args.module} loaded to layers: {lora_layer}")
    net_origin = pkg.Adapter_Sam(copy.deepcopy(sam), lora_layer=lora_layer)

    init_path = os.path.join(output_path, "init.pth")
    if args.rank == 0 and not os.path.exists(init_path):
        copy.deepcopy(net_origin).save_lora_parameters(init_path)
        logger.info(f"Saved initial LoRA checkpoint: {init_path}")
    if args.cuda == -1:
        dist.barrier()

    AIJ = {}
    testloader, test_sampler = {}, {}
    for task_id, key in enumerate(train_data):
        train_data_location = train_data[key]
        test_data_location = test_data[key]
        if args.cuda == -1:
            _, testloader[task_id], _, test_sampler[task_id] = get_dataloader_SAM(
                args, img_embedding_size, train_data_location, test_data_location
            )
        else:
            _, testloader[task_id] = get_dataloader_SAM(
                args, img_embedding_size, train_data_location, test_data_location
            )

    saved_paths = []
    for task_id, key in enumerate(train_data):
        net = copy.deepcopy(net_origin).cuda()
        if args.cuda == -1:
            net = torch.nn.parallel.DistributedDataParallel(
                net, device_ids=[args.rank], find_unused_parameters=True
            )
            net = net.module

        if args.cuda != -1 or args.rank == 0:
            logger.info(f"------Dataset {key} loading begin-------")

        train_data_location = train_data[key]
        test_data_location = test_data[key]
        if args.cuda == -1:
            trainloader, _, train_sampler, _ = get_dataloader_SAM(
                args, img_embedding_size, train_data_location, test_data_location
            )
        else:
            trainloader, _ = get_dataloader_SAM(
                args, img_embedding_size, train_data_location, test_data_location
            )
            train_sampler = None

        save_path = os.path.join(output_path, checkpoint_name(key, args.meth))
        if os.path.exists(save_path):
            if args.rank == 0:
                logger.info(f"{save_path} has been trained.")
            net.load_lora_parameters(save_path, args)
        else:
            if args.meth == "sequ" and task_id > 0:
                net = select(args, net_origin, saved_paths[-1])
                if args.cuda != -1 or args.rank == 0:
                    logger.info(
                        f"---sequential fine tuning on {key} with "
                        f"{os.path.basename(saved_paths[-1])}---"
                    )
            elif args.cuda != -1 or args.rank == 0:
                logger.info(f"---Train {key} model---")

            optimizer, scheduler = configure_opt(
                model=net,
                max_epoch=args.epoch,
                lr=scaled_lr,
                weight_decay=None,
                eta_min=1e-7,
            )
            is_distributed = None
            if args.cuda == -1:
                is_distributed = (train_sampler, test_sampler[task_id])
            train(
                Epoch=args.epoch,
                model=net,
                optimizer=optimizer,
                scheduler=scheduler,
                train_dataloader=trainloader,
                test_dataloader=testloader[task_id],
                logging=logger,
                output_path=output_path,
                args=args,
                is_distributed=is_distributed,
            )
            if args.rank == 0:
                net.save_lora_parameters(save_path)
                logger.info(f"Saved trained LoRA checkpoint: {save_path}")

        if args.cuda == -1:
            dist.barrier()
        saved_paths.append(save_path)

        if args.cuda != -1 or args.rank == 0:
            logger.info(f"---Validation on {save_path}---")
        results = validate_checkpoint(
            args,
            sam,
            net_origin,
            save_path,
            testloader,
            task_id,
            id_to_key,
            logger,
        )
        for index, scores in results.items():
            AIJ[(task_id, index)] = scores
        if args.rank == 0 or args.cuda != -1:
            train_regcl_lib.get_forgetting_metric(task_id, AIJ, logger)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, default="AugModule", choices=["AugModule", "LoRA"], help="Module (Default=AugModule)")
    parser.add_argument("--batch_size", type=int, default=2, help="batch_size per gpu (Default=2)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (Default=0.005)")
    parser.add_argument("--epoch", type=int, default=20, help="Epoch (Default=20)")
    parser.add_argument("--layers", type=str, default="7-11", help="LoRA layers (Default=7-11)")
    parser.add_argument("--vit_name", type=str, default="vit_b", help="select one vit model (Default=vit_b)")
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoint" / "sam_vit_b_01ec64.pth"), help="Pretrained checkpoint")
    parser.add_argument("--img_size", type=int, default=1024, help="input patch size of network input (Default=1024)")
    parser.add_argument("--seed", type=int, default=1234, help="random seed (Default=1234)")
    parser.add_argument("--order", type=str, default="Kvasir_camo_ISTD_ISIC_cod", help="Training order")
    parser.add_argument("--cuda", type=int, default=-1, help="ID of GPU when using single GPU")
    parser.add_argument(
        "--meth",
        type=str,
        default="sequ",
        choices=["indi", "sequ"],
        help="Training mode: independent checkpoints or sequential finetune checkpoints.",
    )
    args = parser.parse_args()
    args.alpha = 1
    args.beta = 10
    args.deta = 1
    args.begin = int(args.layers.split("-")[0])
    args.end = int(args.layers.split("-")[1])
    return args


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)

    output_path = build_output_path(args)
    os.makedirs(output_path, exist_ok=True)
    rank_for_log = int(os.environ.get("RANK", "0"))
    if rank_for_log == 0 and os.path.exists(os.path.join(output_path, "log.txt")):
        open(os.path.join(output_path, "log.txt"), "w").close()
    logging.basicConfig(
        filename=os.path.join(output_path, "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    setup_distribution(args)
    logging.info(str(args))
    train_data, test_data = load_dataset(args)
    train_base_models(args, train_data, test_data, output_path, logging)
