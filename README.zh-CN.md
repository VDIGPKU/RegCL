# RegCL

中文 | [English](readme.md)

## 实现

### 环境

```bash
conda create -n regcl python=3.9.19
conda activate regcl
pip install -r requirements.txt
```

如果 PyTorch 安装失败，可以先安装官方 CUDA 11.6 版本：

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

或者：

```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

### 数据集

所有训练数据集需要整理成如下格式。这里以 Kvasir-SEG 为例：

```text
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

请从各数据集官方网站下载原始数据，并转换为上述格式。

`points.json` 保存每个 instance 的固定 point prompt，主文实验使用这些静态点。

默认数据集配置文件为 `datasets/datasets_train.json` 和 `datasets/datasets_test.json`。当前配置默认使用 `data/` 下的仓库相对路径。请从仓库根目录运行命令，保证这些相对路径可以正确解析；也可以直接把 JSON 文件中的路径改成自己的绝对路径。

### Checkpoint

请下载官方 ViT-B SAM checkpoint 到 `checkpoint/sam_vit_b_01ec64.pth`：

```bash
wget -P checkpoint https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

`checkpoint/` 中已经提供了用于简单测试的 AugModule 预训练参数。提供的 checkpoint 对应如下训练顺序：

```text
Kvasir -> camo -> ISTD -> ISIC -> cod
```

`<dataset_name>.pth` 表示在 `<dataset_name>` 上训练得到的参数。`merged_<dataset_name>.pth` 表示在上述顺序中融合到 `<dataset_name>` 为止的全部知识后的参数。因此在这个顺序下，`merged_cod.pth` 是最后一步融合完成的模型参数。

### 脚本说明

- `train_regcl.py`：本文核心方法的训练脚本，用于训练 AugModule 并逐步进行参数融合。
- `train_module.py`：用于训练独立模块，不进行逐步融合。
- `train_comp_mag.py`：用于训练 RegCL 之外的 merging baseline，是针对对比 merging 方法写的训练脚本。
- `train_replay.py`：用于 replay method 的对比实验。
- `run_experiments.py`：用于集中运行多 seed、多数据顺序、多 layer 设置的实验。

### 使用

- 简单测试

```bash
python test.py --module AugModule --cuda 0 --lora_path checkpoint/merged_cod.pth
```

- 训练 RegCL

```bash
# 分布式训练
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 --master_port=2412 train_regcl.py --module AugModule --batch_size 2 --cuda -1

# 单卡训练
python train_regcl.py --module AugModule --batch_size 8 --cuda 0
```

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@misc{shu2025regclcontinualadaptationsegment,
      title={RegCL: Continual Adaptation of Segment Anything Model via Model Merging}, 
      author={Yuan-Chen Shu and Zhiwei Lin and Yongtao Wang},
      year={2025},
      eprint={2507.12297},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2507.12297}, 
}

## 致谢

感谢 [INV-WZQ/SAMCL](https://github.com/INV-WZQ/SAMCL) 和 [bloomberg/dataless-model-merging](https://github.com/bloomberg/dataless-model-merging) 为本项目提供了有益基础。同时感谢 [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) 提供 Segment Anything Model 及其预训练 checkpoint 基础，也感谢 [tanganke/fusion_bench](https://github.com/tanganke/fusion_bench) 集中提供了 model merging methods 的轻量级实现。

## 许可证
除非另有明确说明，本工具包代码根据知识共享署名-非商业性使用-相同方式共享 4.0 国际公共许可协议 (CC BY-NC-SA 4.0) 的条款提供给您。
该协议包含此处列出的附加条款。
当您从本网站或其他来源下载或使用代码时，
即表示您同意遵守 CC BY-NC-SA 4.0 的条款。
本工具包代码仅用于非商业用途，例如学术研究、教学或科学出版物。
如需商务合作，请联系 wyt@pku.edu.cn。
