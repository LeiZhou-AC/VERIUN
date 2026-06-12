# VERIUN

Machine unlearning verification project scaffold with an ODR runnable pipeline.

## Data Split Modes

`configs/data/dataset.py` supports two modes for `D_u` and `D_r`:

- `split_mode: random`
  Uses `forget_ratio` or `forget_count` for random split.
- `split_mode: by_class`
  Uses `forget_classes` to forget one or more class labels.

Example config:

```yaml
split_mode: by_class
forget_classes: [3, 5]
```

## ODR Test

Random split:

```bash
python ODR_test.py --dataset cifar10 --model-name resnet18 --split-mode random --forget-ratio 0.1
```

Class-based split:

```bash
python ODR_test.py --dataset cifar10 --model-name resnet18 --split-mode by_class --forget-classes 3,5
```

## SCRUB Test

SCRUB is an approximate target-unlearning baseline. It keeps the original model
as a frozen teacher, initializes a student from the same checkpoint, maximizes
forget-set loss and teacher-student disagreement on `D_u`, and preserves utility
on `D_r` with cross-entropy plus distillation.

Random sample-level split using an existing manifest:

```bash
python SCRUB_test.py \
  --dataset cifar10 \
  --model-name resnet18 \
  --split-mode random \
  --forget-ratio 0.01 \
  --forget-manifest-mode load \
  --forget-manifest-path save/manifests/default_forget_manifest.json
```

Class-level split:

```bash
python SCRUB_test.py \
  --dataset cifar10 \
  --model-name resnet18 \
  --split-mode by_class \
  --forget-classes 0
```

After SCRUB finishes, pass the saved checkpoint to RUV:

```bash
python RUV_test.py \
  --unlearned-path save/weights/unlearned/<scrub_checkpoint>.pt \
  --forget-manifest-mode load \
  --forget-manifest-path save/manifests/default_forget_manifest.json
```
