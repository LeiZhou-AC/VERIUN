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

## Unified Unlearning Entry

All implemented unlearning methods can also be launched through the common
factory entry:

```bash
python scripts/unlearn.py \
  --method ssd \
  --dataset cifar10 \
  --model-name resnet18 \
  --split-mode random \
  --forget-ratio 0.01 \
  --forget-manifest-mode load \
  --forget-manifest-path save/manifests/default_forget_manifest.json \
  --trained-path save/weights/trained/resnet18_cifar10.pt \
  --unlearned-path save/weights/unlearned
```

Supported internal methods: `odr`, `odr_gate`, `retrain`, `amnesiac`, `ssd`.
Official SalUn is run through `tools/run_salun_official.py` instead of the
internal unlearning factory.

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

## Official SalUn Workflow

SalUn is evaluated through the official Classification implementation under
`external/salun/Classification`. The project runner keeps the official code
untouched while using this workspace's `datasets/` and `save/` directories.

```bash
python tools/run_salun_official.py \
  --stage all \
  --dataset cifar10 \
  --arch resnet18 \
  --data-path datasets \
  --gpu 0 \
  --seed 2 \
  --train-seed 1 \
  --batch-size 256 \
  --num-indexes-to-replace 4500 \
  --train-epochs 182 \
  --train-lr 0.1 \
  --unlearn-epochs 10 \
  --unlearn-lr 0.013 \
  --mask-ratio 0.5
```

Verify the official SalUn checkpoint with the SalUn backend:

```bash
python RUV_test.py \
  --model-backend salun_official \
  --model-name resnet18 \
  --dataset cifar10 \
  --data-path datasets \
  --split-mode random \
  --forget-manifest-mode load \
  --forget-manifest-path save/manifests/salun_official_random_4500.json \
  --original-model-path save/weights/trained/salun_official_original/0checkpoint.pth.tar \
  --unlearned-model-path save/weights/unlearned/salun_official_random4500/RLcheckpoint.pth.tar \
  --ruv-metric multi_audit \
  --ruv-mode sample
```

## SSD Test

SSD is the post-hoc parameter-importance baseline. It estimates squared-gradient
importance on `D_u` and `D_all`, selects parameters where forget importance
dominates retained importance, and dampens those parameters directly.

```bash
python SSD_test.py \
  --dataset cifar10 \
  --model-name resnet18 \
  --split-mode random \
  --forget-ratio 0.01 \
  --forget-manifest-mode load \
  --forget-manifest-path save/manifests/default_forget_manifest.json \
  --selection-weighting 10.0 \
  --dampening-constant 1.0 \
  --original-split all
```

Class-level SSD:

```bash
python SSD_test.py \
  --dataset cifar10 \
  --model-name resnet18 \
  --split-mode by_class \
  --forget-classes 0
```

ARS activation-route verification can be selected with:

```bash
python RUV_test.py \
  --ruv-metric ars \
  --ruv-mode sample \
  --ruv-layers stem,early,middle \
  --ruv-control-layers late \
  --unlearned-model-path save/weights/unlearned/<checkpoint>.pt \
  --forget-manifest-mode load \
  --forget-manifest-path save/manifests/default_forget_manifest.json
```

ARF adversarial representation fragility verification can be selected with:

```bash
python RUV_test.py \
  --ruv-metric arf \
  --ruv-mode sample \
  --ruv-layers middle,late \
  --ruv-control-layers stem \
  --arf-epsilon 0.031372549 \
  --arf-step-size 0.007843137 \
  --arf-steps 5 \
  --unlearned-model-path save/weights/unlearned/<checkpoint>.pt \
  --forget-manifest-mode load \
  --forget-manifest-path save/manifests/default_forget_manifest.json
```

Two-model multi-view representation audit combines static geometry and dynamic
transition evidence:

```bash
python RUV_test.py \
  --ruv-metric multi_audit \
  --ruv-mode sample \
  --ruv-layers late \
  --ruv-control-layers stem \
  --multi-metrics shift,ruler_m4,npg,ars \
  --multi-threshold 2 \
  --unlearned-model-path save/weights/unlearned/<checkpoint>.pt \
  --forget-manifest-mode load \
  --forget-manifest-path save/manifests/default_forget_manifest.json
```

`ruler_m4` is an oracle-free retain-manifold percentile-rank diagnostic, and
`npg` tracks whether fixed retained neighbours from the original model remain
neighbours after unlearning.
