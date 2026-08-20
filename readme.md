# RegCL

[中文](README.zh-CN.md) | English

## Implementation

### Environment
```
conda create -n regcl python=3.9.19
conda activate regcl
pip install -r requirements.txt
```

If you have trouble installing PyTorch, install the official CUDA 11.6 build first:

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

or:

```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

### Datasets
All training datasets should be stored in the following format. Kvasir-SEG is used here as an example:

```
data/Kvasir/
├── test/
    ├── images/
        ├── 0.png
        ...

    ├── masks/
        ├── points.json
        ├── 0.png
        ...
├── train
    ├── images/
        ├── 0.png
        ...

    ├── masks/
        ├── points.json
        ├── 0.png
        ...
```

Download each dataset from its official website and convert it to the format above.

`points.json` stores the static point prompts for each instance, which are used in the main paper.

The default dataset config files are `datasets/datasets_train.json` and `datasets/datasets_test.json`. They use repository-relative paths under `data/` by default. You can either place the processed datasets under `data/` or edit these JSON files to use your own absolute paths. Relative paths in these config files are resolved from the repository root.

### Checkpoint
Download the official ViT-B SAM checkpoint to `checkpoint/sam_vit_b_01ec64.pth`:

```
wget -P checkpoint https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Pre-trained AugModule checkpoints are provided in `checkpoint/` for simple testing. The provided checkpoints follow this training order:

```
Kvasir -> camo -> ISTD -> ISIC -> cod
```

`<dataset_name>.pth` stores the parameters trained on `<dataset_name>`. `merged_<dataset_name>.pth` stores the merged parameters after integrating all datasets up to `<dataset_name>` in the sequence above. For this order, `merged_cod.pth` is the final merged model checkpoint.

### Scripts

- `train_regcl.py`: the core training script for the proposed method. It trains AugModule and progressively merges the learned parameters.
- `train_module.py`: trains an independent module without progressive merging.
- `train_comp_mag.py`: trains the non-RegCL merging baselines; it is written for the comparison merging methods.
- `train_replay.py`: runs the replay-based comparison method.
- `run_experiments.py`: batch launcher for experiments over multiple seeds, dataset orders, and layer settings.

### Usage

- Simple Testing

```
python test.py --module AugModule --cuda 0 --lora_path checkpoint/merged_cod.pth
```

- Training RegCL

```
# Distributed Training
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 --master_port=2412 train_regcl.py --module AugModule --batch_size 2 --cuda -1

# Single GPU
python train_regcl.py --module AugModule --batch_size 8 --cuda 0
```
