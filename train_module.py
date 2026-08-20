import os
import copy
import random
import logging
import argparse
import numpy as np
from importlib import import_module

import torch
from torch import optim
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
from segment_anything import sam_model_registry
from datasets.dataset import SAM_dataset, RandomGenerator, PairedImageFolders
from datasets.config import load_train_test_configs
from augmodule_utils.utils import train, configure_opt, CustomDataset, MLP, train_embed, AverageMeter, mae, select

from pathlib import Path
from typing import Dict, Any, Tuple
from merging.inner_product import *

ROOT = Path(__file__).parent

def setup_seed(seed: int = 1234) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

def setup_distribution(args: argparse.Namespace) -> None:
    """Initialize the distributed training environment."""
    if args.cuda == -1 and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Distributed training mode
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

        logging.info(f"Initializing rank {args.rank} in {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
        torch.cuda.set_device(args.gpu)

        args.dist_url = 'env://'
        args.dist_backend = 'nccl'
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
        dist.barrier()
    else:
        # Single-GPU training mode
        if args.cuda >= 0:
            torch.cuda.set_device(args.cuda)
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            logging.info(f"Using GPU {current_device}: {device_name}")
        args.rank = 0
        logging.info('Running in single GPU mode')

def load_dataset(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load and initialize datasets."""
    try:
        train_data, test_data = load_train_test_configs(
            ROOT / 'datasets',
            order=args.order,
            repo_root=ROOT,
        )
    except Exception as e:
        logging.error(f"Failed to load dataset: {str(e)}")
        raise

    if args.rank == 0:
        logging.info(f'Dataset order: {list(train_data.keys())}')

    return train_data, test_data

def create_dataloader(dataset, batch_size, sampler=None, shuffle=False):
    if sampler:
        batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=False)
        return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=8, pin_memory=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16, pin_memory=True)

def get_dataloader_SAM(args, img_embedding_size, train_data_location, test_data_location):
    low_res = img_embedding_size * 4
    transform = transforms.Compose([
        RandomGenerator(output_size=[1024, 1024],
                        low_res=[low_res, low_res],
                        bbox_shift=20,
                        get_point=3)
    ])

    db_train = SAM_dataset(train_data_location, transform, 1024, 'train')
    db_test = SAM_dataset(test_data_location, transform, 1024, 'test')
    if args.rank == 0:
            train_dataset_name = train_data_location.split('/')[-2]  # Get dataset name
            test_dataset_name = test_data_location.split('/')[-2]
            logging.info(f"The length of train set ({train_dataset_name}) is: {len(db_train)}")
            logging.info(f"The length of test set ({test_dataset_name}) is: {len(db_test)}\n")

    if args.cuda == -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(db_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(db_test)
        trainloader = create_dataloader(db_train, args.batch_size, sampler=train_sampler)
        testloader = create_dataloader(db_test, 1, sampler=test_sampler)
        return trainloader, testloader, train_sampler, test_sampler
    else:
        trainloader = create_dataloader(db_train, args.batch_size, shuffle=True)
        testloader = create_dataloader(db_test, 1, shuffle=True)
        return trainloader, testloader

def get_forgetting_metric(lenth, AIJ, logging):
    AA = [0., 0., 0.]
    FM = [0., 0., 0.]
    FT = [0., 0., 0.]
    for id in range(lenth+1):
        AA[0]+=AIJ[(lenth, id)][0]/(lenth+1)
        AA[1]+=AIJ[(lenth, id)][1]/(lenth+1)
        AA[2]+=AIJ[(lenth, id)][2]/(lenth+1)
        FM[0]+=(AIJ[(id, id)][0] - AIJ[(lenth, id)][0])/(lenth+1)
        FM[1]+=(AIJ[(id, id)][1] - AIJ[(lenth, id)][1])/(lenth+1)
        FM[2]+=(AIJ[(id, id)][2] - AIJ[(lenth, id)][2])/(lenth+1)
        if id<lenth:
            FT[0]+=AIJ[(id, id+1)][0]/lenth
            FT[1]+=AIJ[(id, id+1)][1]/lenth
            FT[2]+=AIJ[(id, id+1)][2]/lenth

    for i in range(len(AA)):
        AA[i] = AA[i].cpu().item()
        FM[i] = FM[i].cpu().item()
        if torch.is_tensor(FT[i]):
            FT[i] = FT[i].cpu().item()
    if args.rank==0:
        logging.info("AA: {}".format(AA))
        logging.info("FM: {}".format(FM))
        logging.info("FT: {}".format(FT))

def main(args, train_data, test_data, output_path, logging):
    # ---------------Initialization---------------
    key_to_id = {key: id for id, key in enumerate(train_data)}
    id_to_key = {id: key for id, key in enumerate(train_data)}
    order_lst = list(train_data.keys())
    # lr scaling
    world_size = args.world_size if args.cuda == -1 else 1
    base_lr = args.lr  # 0.005 for total batchsize 8
    scaled_lr = base_lr * (world_size / 4.0)  # scale lr based on world size

    if args.rank == 0:
        logging.info(f"lr scaling: base_lr={base_lr}, gpu_nums={world_size}, actual_lr={scaled_lr}")

    lora_layer = list(range(args.begin, args.end + 1))

    # Load the SAM model and freeze image encoder parameters
    sam, img_embedding_size = sam_model_registry[args.vit_name](checkpoint=args.ckpt)
    sam = sam.cuda()
    for name, param in sam.image_encoder.named_parameters():
        param.requires_grad = False
    sam.image_encoder.train(mode=False)

    pkg = import_module(f'module.{args.module}')
    logging.info(f"Module {args.module} loaded to layers: {lora_layer}")
    net_origin = pkg.Adapter_Sam(copy.deepcopy(sam), lora_layer=lora_layer)

    AIJ = {}
    testloader, test_sampler = {}, {}

    # ---------------load all tasks' testloader---------------
    for id, key in enumerate(train_data):
        test_data_location = test_data[key]
        train_data_location= train_data[key]
        if args.cuda==-1:
            _, testloader[id], _, test_sampler[id] = get_dataloader_SAM(args, img_embedding_size, train_data_location, test_data_location)
        else:
            _, testloader[id] = get_dataloader_SAM(args, img_embedding_size, train_data_location, test_data_location)

    # ---------------Trian LoRA models---------------
    for id, key in enumerate(train_data):
        net = copy.deepcopy(net_origin).cuda()
        if args.cuda==-1:
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.rank], find_unused_parameters=True)
            net = net.module
        if args.cuda!=-1 or args.rank==0:
            logging.info(f"------Dataset {key} loading begin-------")

        test_data_location = test_data[key]
        train_data_location= train_data[key]
        if args.cuda==-1:
            trainloader, _, train_sampler, _ = get_dataloader_SAM(args, img_embedding_size, train_data_location, test_data_location)
        else:
            trainloader, testloader[id] = get_dataloader_SAM(args, img_embedding_size, train_data_location, test_data_location)

        if os.path.exists(f'{output_path}/{key}.pth'):
            if args.rank==0:
                logging.info(f"{key} model has been trained.")
            net.load_lora_parameters(f'{output_path}/{key}.pth', args)
        else:
            if args.meth == 'indi' or id == 0:
                # Train LoRA
                if args.cuda!=-1 or args.rank==0:
                    logging.info(f'---Train {key} model---')
            elif args.meth == 'sequ' and id > 0:
                net = select(args, net_origin, f'{output_path}/{order_lst[id-1]}.pth')
                if args.cuda!=-1 or args.rank==0:
                    logging.info(f'---sequential fine tuning on {key} with {order_lst[id-1]}.pth---')
            else:
                raise NotImplementedError

            # Configure the optimizer and learning rate scheduler
            optimizer, scheduler = configure_opt(
                model=net,
                max_epoch=args.epoch,
                lr=scaled_lr,
                weight_decay=None,
                eta_min=1e-7
            )
            is_distributed=None
            if args.cuda==-1:
                is_distributed=(train_sampler, test_sampler[id])
            train(Epoch=args.epoch,
                model=net,
                optimizer=optimizer,
                scheduler=scheduler,
                train_dataloader=trainloader,
                test_dataloader=testloader[id],
                logging=logging,
                output_path=output_path,
                args=args,
                is_distributed=is_distributed
                )
            net.save_lora_parameters(f'{output_path}/{key}.pth')

        if args.cuda == -1:
            dist.barrier()

        # ---------------Validation---------------
        if args.cuda!=-1 or args.rank==0:
            logging.info(f"---Validation on model {key}---")

        val_net = select(args, net_origin, f'{output_path}/{key}.pth')
        val_net.eval()

        for index in range(len(train_data)): # Evaluate on all datasets, previously min(len(train_data), id+2)
            ious = AverageMeter()
            f1_scores = AverageMeter()
            mae_scores = AverageMeter()
            if args.cuda!=-1 or args.rank==0:
                logging.info(f'-----{index} of {id_to_key[index]} begin test-------')
            for iter, data in enumerate(testloader[index]):
                images, gt_masks, points = data["image"].cuda(non_blocking=True), data["label"].cuda(non_blocking=True), data['point']
                data['point'][0], data['point'][1] = data['point'][0].cuda(non_blocking=True), data['point'][1].cuda(non_blocking=True)
                with torch.no_grad():
                    if args.begin > 0:
                        end_before_lora = args.begin - 1
                        input_images = sam.preprocess(images)
                        mid_embed, embed = sam.image_encoder(input_images, False, begin=-1, end=end_before_lora)
                        outputs = val_net(mid_embed, points=points, begin=args.begin, end=-1)
                    else:
                        outputs = val_net(images, points=points)
                for image_, pred_mask, gt_mask in zip(images, outputs["masks"], gt_masks):
                    if len(gt_mask.size())<3:
                        gt_mask = gt_mask.unsqueeze(0)
                    batch_stats = smp.metrics.get_stats(
                        torch.sigmoid(pred_mask),
                        gt_mask.int(),
                        mode='binary',
                        threshold=0.5,
                    )
                    batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                    batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                    batch_mae = mae(pred_mask, gt_mask)
                    if args.cuda==-1:
                        iou_list = [torch.zeros_like(batch_iou.clone().detach()) for _ in range(dist.get_world_size())]
                        f1_list = [torch.zeros_like(batch_f1.clone().detach()) for _ in range(dist.get_world_size())]
                        mae_list = [torch.zeros_like(batch_mae.clone().detach()) for _ in range(dist.get_world_size())]
                        dist.all_gather(iou_list, batch_iou)
                        dist.all_gather(f1_list, batch_f1)
                        dist.all_gather(mae_list, batch_mae)
                        for lenth in range(len(iou_list)):
                            mae_scores.update(mae_list[lenth], 1)
                            ious.update(iou_list[lenth], 1)
                            f1_scores.update(f1_list[lenth], 1)
                    else:
                        mae_scores.update(batch_mae, 1)
                        ious.update(batch_iou, 1)
                        f1_scores.update(batch_f1, 1)
                if logging!=None and iter%50==0:
                    if args.cuda!=-1 or args.rank==0:
                        logging.info(
                            f'Val: [[{iter}/{len(testloader[index])}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] -- MAE: [{mae_scores.avg:.4f}]'
                        )
            if logging!=None:
                if args.rank==0 or args.cuda!=-1:
                    logging.info(
                        f'Val: [[{iter}/{len(testloader[index])}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}] -- MAE: [{mae_scores.avg:.4f}]'
                    )
            if index <= id + 1: AIJ[(id,index)] = (ious.avg, f1_scores.avg, mae_scores.avg)
        if args.rank==0 or args.cuda!=-1: get_forgetting_metric(id, AIJ, logging)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str,
                        default='AugModule', choices=['AugModule', 'LoRA'], help='Module (Default=AugModule)')
    parser.add_argument('--num_embedding', type=int,
                        default=300, help='Number of stored embedding per domain (default=300)')
    parser.add_argument('--batch_size', type=int,
                        default=2, help='batch_size per gpu (Default=2)')
    parser.add_argument('--lr', type=float,
                        default=0.005, help='Learning rate (Default=0.005)')
    parser.add_argument('--epoch', type=int,
                        default=20, help='Epoch (Default=20)')
    parser.add_argument('--layers', type=str,
                        default='7-11', help='LoRA layers (Default=7-11)')
    parser.add_argument('--vit_name', type=str,
                        default='vit_b', help='select one vit model (Default=vit_b)')
    parser.add_argument('--ckpt', type=str,
                        default=str(ROOT / 'checkpoint' / 'sam_vit_b_01ec64.pth'), help='Pretrained checkpoint')
    parser.add_argument('--img_size', type=int,
                        default=1024, help='input patch size of network input (Default=1024)')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed (Default=1024)')
    parser.add_argument('--order', type=str,
                        default="Kvasir_camo_ISTD_ISIC_cod", help="Training order (Default=Kvasir_camo_ISTD_ISIC_cod)")
    parser.add_argument('--cuda', type=int,
                        default=-1, help='ID of GPU when using single GPU (cuda=-1 means using distributed GPU)')
    parser.add_argument('--meth', type=str,
                        default='indi', choices=['indi', 'sequ'], help='Training mode: indi or sequ')
    args = parser.parse_args()
    args.alpha = 1
    args.beta = 10
    args.deta = 1

    args.begin = int(args.layers.split('-')[0])
    args.end = int(args.layers.split('-')[1])

    output_path = f'log/Comparison/{args.module}/{args.meth}_{args.begin}_{args.end}_4_0005'
    # output_path = f'log/Comparison/LoRA/{args.module}_{args.begin}_{args.end}_4_0005_L1'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if os.path.exists(f'{output_path}/log.txt'):
        open(f'{output_path}/log.txt', 'w').close()
    logging.basicConfig(filename=f'{output_path}/log.txt', level=logging.INFO,
                        format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(str(args))

    setup_seed(args.seed)
    setup_distribution(args)
    train_data, test_data = load_dataset(args)

    main(args, train_data, test_data, output_path, logging)
