# vMFcache

Cache-based DQDA + vMF mixture online test-time adaptation for fine-grained vision datasets.

## Setup

```bash
pip install -r requirements.txt
```

Symlink pre-extracted class text embeddings (from ADAPT or your own extraction):

```bash
ln -s /path/to/ADAPT/pre_extracted_class_feat ./pre_extracted_class_feat
```

## Quick start

```bash
bash scripts/run_eurosat.sh [GPU_ID]
```

Supported datasets: `fgvc_aircraft`, `caltech101`, `stanford_cars`, `dtd`, `eurosat`, `oxford_flowers`, `food101`, `oxford_pets`, `sun397`, `ucf101`.

## Main entry

```bash
python vMFcache.py --data /path/to/datasets --test_set eurosat --arch ViT-B/16 --gpu 0
```
