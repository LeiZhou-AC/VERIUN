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
