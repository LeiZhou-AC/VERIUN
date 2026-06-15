# VERIUN

Machine unlearning verification project scaffold with an ODR runnable pipeline.

## Data Split Modes

`configs/data/dataset.py` supports the main unlearning granularities used in
the experiments:

- `split_mode: random`
  Arbitrary random sample-level split over the full training set.
- `split_mode: by_class`
  Class-level split. All samples from `forget_classes` are assigned to `D_u`.
- `split_mode: class_random`
  Class-conditioned sample-level split. Samples are randomly selected inside
  each class listed in `forget_classes`.

For `class_random`, `forget_ratio` is applied per selected class. If
`forget_count` is provided, it is also interpreted per selected class. Thus,
`forget_classes: [0]` gives single-class conditioned sample-level unlearning,
while `forget_classes: [0, 3, 5]` gives multi-class conditioned sample-level
unlearning.

Example config:

```yaml
split_mode: by_class
forget_classes: [3, 5]
```

Class-conditioned sample-level example:

```yaml
split_mode: class_random
forget_classes: [0]
forget_ratio: 0.1
forget_manifest_mode: save
forget_manifest_path: save/manifests/cifar10_class0_ratio0p1.json
```

Multi-class conditioned sample-level example:

```yaml
split_mode: class_random
forget_classes: [0, 3, 5]
forget_count: 100
forget_manifest_mode: save
forget_manifest_path: save/manifests/cifar10_classes0-3-5_count100.json
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

Class-conditioned sample split:

```bash
python ODR_test.py --dataset cifar10 --model-name resnet18 --split-mode class_random --forget-classes 0 --forget-ratio 0.1 --forget-manifest-mode save
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
