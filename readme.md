# NSLoRA

Official code release for **TODO: paper title**.

This repository contains the training and evaluation code for NSLoRA-style continual learning experiments on CIFAR-100, ImageNet-R, ImageNet-100, DomainNet, and CUB-200. Paper metadata and BibTeX are left as placeholders and should be updated before the final public release.

## Environment

The experiments were developed with Python 3.11 and PyTorch 2.1.0. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Pretrained ViT and CLIP weights are loaded through `timm`, `open_clip_torch`, and the local `clip` module. If you need a custom Hugging Face endpoint or offline cache, configure it in your shell before running training, for example:

```bash
export HF_ENDPOINT=https://huggingface.co
export HF_HUB_CACHE=/path/to/cache
```

## Dataset Preparation

Download the datasets from their official sources:

- CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
- ImageNet-R: https://github.com/hendrycks/imagenet-r
- DomainNet: https://ai.bu.edu/M3SDA/
- CUB-200: configure in ImageFolder-style `train/val` folders
- ImageNet-100: prepare from ImageNet with the provided split files

The training code expects this layout:

```text
DATA_ROOT/
  train/
    class_1/
      image_1.jpg
      image_2.jpg
    class_2/
      image_3.jpg
  val/
    class_1/
      image_4.jpg
    class_2/
      image_5.jpg
```

Split files and conversion utilities are in `tools/`. The tools create symlinks by default; pass `--copy` if symlinks are not available on your platform.

```bash
python tools/split_cifar100.py --source-root /path/to/raw_cifar100 --output-root /path/to/prepared_cifar100
python tools/split_imagenet_r.py --source-root /path/to/raw_imagenet_r --output-root /path/to/prepared_imagenet_r
python tools/split_imagenet100.py --source-root /path/to/imagenet --output-root /path/to/prepared_imagenet100
python tools/split_sdomainnet.py --source-root /path/to/raw_domainnet --output-root /path/to/prepared_domainnet
```

## Training

All experiments use `train_eval.py`:

```bash
python train_eval.py -d <dataset> --data_root <DATA_ROOT> [options]
```

Supported dataset names are:

- `cifar100`
- `imagenet_r`
- `imagenet100`
- `sdomainet`
- `cub200`

The root scripts provide the main reproduction commands. Set the dataset root with a dataset-specific environment variable or with `DATA_ROOT`.

```bash
CIFAR100_ROOT=/path/to/prepared_cifar100 bash train_cifar100_s10_clip.sh
CIFAR100_ROOT=/path/to/prepared_cifar100 bash train_cifar100_s20_clip.sh
IMAGENET_R_ROOT=/path/to/prepared_imagenet_r bash train_imagenet_r_s10_clip.sh
IMAGENET_R_ROOT=/path/to/prepared_imagenet_r bash train_imagenet_r_s20_clip.sh
DOMAINNET_ROOT=/path/to/prepared_domainnet bash train_domainnet_clip.sh
CUB200_ROOT=/path/to/prepared_cub200 bash train_cub200_clip.sh
```

VPT baselines are also provided:

```bash
CIFAR100_ROOT=/path/to/prepared_cifar100 bash train_cifar100_s10_vpt.sh
CIFAR100_ROOT=/path/to/prepared_cifar100 bash train_cifar100_s20_vpt.sh
IMAGENET_R_ROOT=/path/to/prepared_imagenet_r bash train_imagenet_r_s10_vpt.sh
IMAGENET_R_ROOT=/path/to/prepared_imagenet_r bash train_imagenet_r_s20_vpt.sh
DOMAINNET_ROOT=/path/to/prepared_domainnet bash train_domainnet_vpt.sh
```

Training checkpoints are written to `check_point/`, which is ignored by Git.

## Repository Layout

```text
clip/          CLIP model and tokenizer utilities
tools/         Dataset split lists and preparation utilities
utils/         Dataset, continual learning, optimizer, and ViT helpers
train_eval.py  Main training and evaluation entry point
*.sh           Reproduction scripts
```

## Citation

```bibtex
@inproceedings{TODO,
  title     = {TODO},
  author    = {TODO},
  booktitle = {TODO},
  year      = {TODO}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
