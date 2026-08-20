import os
import json
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
from augmodule_utils.utils import train, configure_opt, CustomDataset, MLP, train_embed, AverageMeter, mae, select

from typing import Dict, Any, Tuple
from merging.inner_product import *

class Combined_SAM_set(SAM_dataset):
    def __init__(self, Fulldata_loca, Sampdata_loca_lst: list, transform=None, inp_size=1024, type='train', sample_size=300):
        super().__init__(Fulldata_loca, transform, inp_size, type)

        # Load full dataset
        Full_set = PairedImageFolders(Fulldata_loca+"/images", Fulldata_loca+"/masks")

        sampled_set = []
        for loca in Sampdata_loca_lst:
            samp_set = PairedImageFolders(loca+"/images", loca+"/masks")
            indices = torch.randperm(len(samp_set))[:sample_size]
            sampled_set.extend([samp_set[i] for i in indices])

        # Combine datasets
        self.dataset = Full_set + sampled_set
        self.inp_size = inp_size
        self.type = type
        self.transform = transform

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
    # Load dataset config files
    try:
        with open('datasets/datasets_test.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        with open('datasets/datasets_train.json', 'r', encoding='utf-8') as f:
            train_data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load dataset: {str(e)}")
        raise

    # Reorder datasets according to the training order
    keys = args.order.split("_")
    train_data = {key: train_data[key] for key in keys}
    test_data = {key: test_data[key] for key in keys}

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

def get_dataloader_combined(args, img_embedding_size, fullset_path, sampset_paths: list, samlpe_size=300):
    low_res = img_embedding_size * 4
    transform = transforms.Compose([
        RandomGenerator(output_size=[1024, 1024],
                        low_res=[low_res, low_res],
                        bbox_shift=20,
                        get_point=3)
    ])

    db_train = Combined_SAM_set(fullset_path, sampset_paths, transform, 1024, 'train', sample_size=samlpe_size)

    if args.rank == 0:
        logging.info(f"The length of combined train set is: {len(db_train)}")

    if args.cuda == -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(db_train)
        trainloader = create_dataloader(db_train, args.batch_size, sampler=train_sampler)
        return trainloader, train_sampler
    else:
        trainloader = create_dataloader(db_train, args.batch_size, shuffle=True)
        return trainloader

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

    start = int(args.layers.split('-')[0])
    end = int(args.layers.split('-')[1])
    lora_layer = list(range(start, end + 1))

    grams_dict = {}
    filterd_layer_inputs = [f'sam.image_encoder.blocks.{i+start}.attn.B' for i in range(5)] + \
                           [f'sam.image_encoder.blocks.{i+start}.attn.C' for i in range(5)] + \
                           [f'A']
    filterd_layer_params = [
                            '.*extra.*',
                            '.*A.*',
                            '.*\.attn\.norm1.*',
                            '.*\.attn\.norm2.*',
                            '.*\.attn\.C.*',
                            '.*\.attn\.B.*'
                            ]

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

            # Train LoRA
            if args.cuda!=-1 or args.rank==0:
                logging.info(f'---Train {key} model---')
            train(Epoch=args.epoch,
                model=net,
                optimizer=optimizer,
                scheduler=scheduler,
                train_dataloader=trainloader,
                test_dataloader=testloader[id],
                logging=logging,
                output_path=output_path,
                args=args,
                is_distributed=is_distributed)
            net.save_lora_parameters(f'{output_path}/{key}.pth')

        # ---------------Computing gram---------------
        if args.rank==0:
            with torch.no_grad():
                model = copy.deepcopy(net_origin).to('cuda:0')
                model.load_lora_parameters(f'{output_path}/{key}.pth', args)
                logging.info(f'{key} gram computing...\n')
                grams = compute_gram(model, trainloader, filterd_layer_inputs)
                grams_dict.update({key: grams})

    # ---------------Merging LoRA models with replay---------------
    for id, key in enumerate(train_data):
        if id != 0:
            if args.cuda!=-1 or args.rank==0:
                net = copy.deepcopy(net_origin).to('cuda:0')
                logging.info(f"------Merging {key} with previous-------")

            # ---------------loading dataloader for refine---------------
            fullset_loca = train_data[key]
            sampset_loca = [train_data[order_lst[j]] for j in range(id)]
            if args.cuda==-1:
                rep_trainloader, rep_train_sampler = get_dataloader_combined(args, img_embedding_size, fullset_loca, sampset_loca, samlpe_size=args.num_embedding)
            else:
                rep_trainloader = get_dataloader_combined(args, img_embedding_size, fullset_loca, sampset_loca, samlpe_size=args.num_embedding)

            # ---------------loading model_params for merge---------------
            params_name, param_lst, grams_lst = [], [], []
            params_name.append(f'{key}')
            if id == 1:
                params_name.append(f'{id_to_key[id-1]}')
            else:
                params_name.append(f'merged_{id_to_key[id-1]}')

            if args.rank==0:
                with torch.no_grad():
                    for k in params_name: # load_model_prams
                        x = copy.deepcopy(net_origin).to('cuda:0')  # Move the model to the target GPU
                        x.load_lora_parameters(f'{output_path}/{k}.pth', args)
                        logging.info(f'model {k} loaded on {next(x.parameters()).device}')
                        param_lst.append(x)

                with torch.cuda.device(0):
                    grams_lst = [grams_dict[key] for key in params_name]
                    fused_weights = avg_merge(param_lst, grams_lst, filterd_layer_params)
                    copy_params_to_model(fused_weights, net)

            if args.cuda==-1:
                net = torch.nn.parallel.DistributedDataParallel(net.cuda(), device_ids=[args.rank], find_unused_parameters=True)
                net = net.module

            # ---------------refine LoRA model---------------
            optimizer, scheduler = configure_opt(
                model=net,
                max_epoch=args.epoch,
                lr=scaled_lr,
                weight_decay=None,
                eta_min=1e-7
            )
            is_distributed=None
            if args.cuda==-1:
                is_distributed=(rep_train_sampler, test_sampler[id])

            # Train LoRA
            if args.cuda!=-1 or args.rank==0:
                logging.info(f'---Refine {key} and {[order_lst[j] for j in range(id)]} sampled 300 merged model---')
            train(Epoch=args.epoch,
                model=net,
                optimizer=optimizer,
                scheduler=scheduler,
                train_dataloader=rep_trainloader,
                test_dataloader=testloader[id],
                logging=logging,
                output_path=output_path,
                args=args,
                is_distributed=is_distributed)
            net.save_lora_parameters(f'{output_path}/merged_{key}.pth')
            if args.cuda == -1:
                dist.barrier()

            if args.rank==0:
                with torch.no_grad():
                    model = copy.deepcopy(net_origin).to('cuda:0')  # Move the model to the target GPU
                    model.load_lora_parameters(f'{output_path}/merged_{key}.pth', args)
                    logging.info(f'merged {key} gram computing...')
                    grams = compute_gram(model, rep_trainloader, filterd_layer_inputs)
                    grams_dict.update({f'merged_{key}': grams})
            if args.cuda == -1:
                dist.barrier()

        # ---------------Validation---------------
        if args.cuda!=-1 or args.rank==0:
            logging.info(f"---Validation---")

        for index in range(min(len(train_data), id+2)):
            ious = AverageMeter()
            f1_scores = AverageMeter()
            mae_scores = AverageMeter()
            if args.cuda!=-1 or args.rank==0:
                logging.info(f'-----{index} of {id_to_key[index]} begin test-------')
            for iter, data in enumerate(testloader[index]):
                images, gt_masks, points = data["image"].cuda(non_blocking=True), data["label"].cuda(non_blocking=True), data['point']
                data['point'][0], data['point'][1] = data['point'][0].cuda(non_blocking=True), data['point'][1].cuda(non_blocking=True)

                if id == 0:
                    val_net = select(args, net_origin, f'{output_path}/{key}.pth')
                else:
                    val_net = select(args, net_origin, f'{output_path}/merged_{key}.pth')
                val_net.eval()
                with torch.no_grad():
                    if start > 0:
                        end_before_lora = start - 1
                        input_images = sam.preprocess(images)
                        mid_embed, embed = sam.image_encoder(input_images, False, begin=-1, end=end_before_lora)
                        outputs = val_net(mid_embed, points=points, begin=start, end=-1)
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
            AIJ[(id,index)] = (ious.avg, f1_scores.avg, mae_scores.avg)

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
                        default='checkpoint/sam_vit_b_01ec64.pth', help='Pretrained checkpoint')
    parser.add_argument('--img_size', type=int,
                        default=1024, help='input patch size of network input (Default=1024)')

    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed (Default=1024)')
    parser.add_argument('--order', type=str,
                        default="Kvasir_camo_ISTD_ISIC_cod", help="Training order (Default=Kvasir_camo_ISTD_ISIC_cod)")
    parser.add_argument('--cuda', type=int,
                        default=-1, help='ID of GPU when using single GPU (cuda=-1 means using distributed GPU)')
    args = parser.parse_args()

    # output_path = f'log/SAM_LoRAs'
    output_path = f'log/Regplay_AugModule_300_7_11_4_0025'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if os.path.exists(f'{output_path}/log_replay.txt'):
        open(f'{output_path}/log_replay.txt', 'w').close()
    logging.basicConfig(filename=f'{output_path}/log_replay.txt', level=logging.INFO,
                        format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(str(args))

    setup_seed(args.seed)
    setup_distribution(args)
    train_data, test_data = load_dataset(args)

    main(args, train_data, test_data, output_path, logging)
