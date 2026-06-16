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

## Amnesiac Test

Amnesiac is the approximate target-unlearning baseline used in the current
project. The implemented variant is post-hoc friendly: it loads the trained
checkpoint, relabels `D_u` with deterministic wrong labels, continues training
on relabeled `D_u`, and uses a controlled number of `D_r` batches to preserve
utility.

Wrong-label strategies are configurable with `--label-strategy`:

- `cyclic`: maps each class to `(y + 1) mod K` and is the default.
- `permutation`: uses a seeded class derangement.
- `random`: keeps the old per-sample random wrong-label behavior.

Random sample-level split using an existing manifest:

```bash
python AMNESIAC_test.py \
  --dataset cifar10 \
  --model-name resnet18 \
  --split-mode random \
  --forget-ratio 0.01 \
  --forget-manifest-mode load \
  --forget-manifest-path save/manifests/default_forget_manifest.json \
  --label-strategy cyclic
```

Class-level split:

```bash
python AMNESIAC_test.py \
  --dataset cifar10 \
  --model-name resnet18 \
  --split-mode by_class \
  --forget-classes 0
```

Class-conditioned sample-level split:

```bash
python AMNESIAC_test.py \
  --dataset cifar10 \
  --model-name resnet18 \
  --split-mode class_random \
  --forget-classes 0 \
  --forget-ratio 0.1 \
  --forget-manifest-mode load \
  --forget-manifest-path save/manifests/cifar10_class0_ratio0p1.json
```

After Amnesiac finishes, pass the saved checkpoint to RUV:

```bash
python RUV_test.py \
  --unlearned-model-path save/weights/unlearned/<amnesiac_checkpoint>.pt \
  --forget-manifest-mode load \
  --forget-manifest-path save/manifests/default_forget_manifest.json
```
