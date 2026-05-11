# Solution

## Overview

The final solution focuses on initializing the last classification layer of a pretrained ResNet18 model.

The model consists of two parts:

```text
image -> pretrained ResNet18 backbone -> 512-dimensional feature vector -> CIFAR100 head
```

The ImageNet-pretrained backbone already extracts useful visual features, but its original classification head is designed for ImageNet classes. For CIFAR100, we need a new `Linear(512, 100)` head.

Because the task has a strict training budget,

```text
n_batches × batch_size ≤ 8192
```

we avoid expensive full training of the head. Instead, we use the allowed budget to build a strong initialization of the final layer.

The final method uses a balanced subset of the CIFAR100 training split:

```text
81 samples per class × 100 classes = 8100 samples
```

This stays within the limit:

```text
8100 / 8192 samples
```

No validation data is used during initialization.

---

## How We Arrived at This Solution

We started with the most direct approach: use zero-order optimization to tune the final CIFAR100 head. The head has 51,300 parameters, and we tried several SPSA-style update rules to optimize these parameters without gradients.

However, this approach was unstable. Under the budget constraint

```text
n_batches × batch_size ≤ 8192
```

there are too few samples to reliably estimate useful zero-order updates for such a large parameter vector. Early SPSA experiments gave weak results, and later variants produced only marginal improvements.

This led to the main observation: the bottleneck is not the exact zero-order optimizer, but the initial quality of the final classification head.

The ImageNet-pretrained ResNet18 backbone already produces useful visual features. Therefore, instead of trying to discover the CIFAR100 head from scratch with noisy zero-order updates, we decided to construct a strong initial head directly from the training features.

The first version used class prototypes computed from CIFAR100 train features. We passed training images through the frozen backbone and averaged feature vectors within each class. This immediately gave a large improvement over the ImageNet head, which confirmed that the backbone features were already informative for CIFAR100.

Then we refined this idea to make it budget-compliant. Instead of using the full CIFAR100 train split, we used exactly 81 samples per class:

```text
81 × 100 = 8100 samples
```

This fits the allowed budget of 8192 samples and keeps the subset balanced across classes.

Finally, we added two normalization steps:

1. normalize each image feature before averaging;
2. normalize each final class prototype before copying it into `fc.weight`.

This produced the final solution: a balanced, normalized prototype initialization of the final ResNet18 head.

---

## Final Result

The final run produced the following result:

```text
1. Baseline (ImageNet head)       0.37%
2. Initialized head (no FT)      52.77%
3. Fine-tuned (ZO)               52.77%
------------------------------------------------------------
Budget: 81 steps × batch 100 = 8,100 / 8192 samples
Layers tuned: (none)
Val samples:  10,000
```

The main improvement comes from the initialized head. Zero-order fine-tuning was not used in the final configuration, because the initialized head already gives a strong result and direct ZO tuning of the full head was unstable under the limited budget.

---

## Head Initialization Method

The final solution is implemented in `head_init.py`.

The procedure is:

1. Load pretrained ResNet18 with ImageNet weights.
2. Replace the original final classifier with `Identity`.
3. Select exactly 81 training samples from each CIFAR100 class.
4. Forward these 8100 images through the frozen ResNet18 backbone.
5. For each image, obtain a 512-dimensional feature vector.
6. Normalize each individual feature vector.
7. For each class, compute the mean of its normalized feature vectors.
8. Normalize the final class prototype.
9. Copy the 100 class prototypes into `fc.weight`.
10. Set `fc.bias` to zero.

So each row of the final head weight matrix corresponds to one CIFAR100 class prototype:

```text
fc.weight[class_id] = normalized class prototype
```

The final head has shape:

```text
fc.weight: 100 × 512
fc.bias:   100
```

---

## Why This Works

The pretrained ResNet18 backbone maps visually similar images to nearby feature vectors. Therefore, images from the same CIFAR100 class tend to have related feature representations.

For each class, we estimate a representative direction in feature space by averaging normalized feature vectors from that class.

At inference time, the final linear layer computes:

```text
logit_c = feature · prototype_c
```

Since both image features and class prototypes are direction-based, this behaves similarly to cosine similarity. The predicted class is the one whose prototype is most aligned with the image feature.

This gives a strong classifier without gradient-based training.

---

## Why We Normalize Features

Feature normalization is important because we want the class prototype to represent the direction of the class, not the raw magnitude of ResNet features.

The process is:

```text
feature = feature / ||feature||
```

before averaging.

Then the final class prototype is also normalized:

```text
prototype = prototype / ||prototype||
```

This makes classification depend mainly on angular similarity between the image representation and the class prototype.

In experiments, removing prototype normalization significantly reduced accuracy, so normalization was kept in the final method.

---

## Zero-Order Optimization Experiments

We also tested several SPSA-style zero-order optimization variants. The motivation was to improve the initialized head after prototype construction.

However, the final head contains:

```text
512 × 100 + 100 = 51,300 parameters
```

Under the small budget of 8192 samples, direct zero-order optimization of this many parameters gives very noisy estimates.

The tested optimizer variants gave only marginal or unstable improvements. Therefore, the final submitted configuration keeps the prototype-initialized head and does not tune any layers:

```text
Layers tuned: (none)
```

This makes the final result more stable and reproducible.

---

## Experiments

### Early Direction: SPSA

We first tried to directly optimize the classification head using SPSA-style zero-order methods. This direction was not effective enough because the parameter dimension is large and the number of allowed samples is small.

The conclusion was that the quality of the initial head is more important than the exact SPSA variant.

### Prototype Initialization

The strongest stable direction was to initialize the head using class prototypes computed from frozen ResNet18 features.

The final version uses:

```text
81 samples per class
100 classes
8100 total training samples
feature normalization before averaging
final prototype normalization
zero bias
```

This gave:

```text
Initialized head accuracy: 52.77%
Fine-tuned accuracy:       52.77%
```

Since no layers were tuned after initialization, the initialized and fine-tuned accuracies are the same.

---

## Reproducibility

The final result can be reproduced with:

```bash
python validate.py \
  --data_dir ./data \
  --batch_size 100 \
  --n_batches 81 \
  --seed 42 \
  --output results.json
```

The effective budget is:

```text
81 × 100 = 8100 samples
```

which satisfies the task constraint:

```text
8100 ≤ 8192
```

---

## Files Changed

The final solution changes only:

```text
head_init.py
```

The following files are not modified:

```text
validate.py
model.py
```

---

## Conclusion

The final solution is therefore the result of moving from unstable direct zero-order optimization to a more reliable feature-based initialization strategy.

Instead of trying to learn 51,300 head parameters from scratch with noisy zero-order updates, we use the pretrained ResNet18 backbone to extract features from a balanced CIFAR100 subset and build one normalized prototype per class.

This gives a simple, stable, and budget-compliant initialization of the CIFAR100 classification head.

The final achieved validation accuracy is:

```text
52.77%
```