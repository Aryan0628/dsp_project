# Lesson 07 - The Deep Learning Model (ResNet-18)

> **In one minute**
> A neural network is many small number-crunching units (neurons) stacked in layers. A convolutional neural network (CNN) is one that slides small learned filters over an image. We do not train it from zero: we take a ResNet-18 that already learned from ImageNet photos and fine-tune it on our MRI scans (transfer learning). Training repeats one loop: guess, measure how wrong, work out which weights to nudge, nudge them. We use it for the hard question "which kind of tumor?", and blend its answer with the classical model.

## What you'll learn

- What a neuron and a layer are, and how stacking them gives a neural network.
- What a CNN is, and how its filters differ from the hand-designed filters of Lesson 03.
- Pooling, softmax, cross-entropy loss, gradient descent and backpropagation, each in plain terms.
- Learning rate, AdamW, the cosine schedule, label smoothing, class weights and test-time augmentation, each in plain words.
- Epochs and batches, transfer learning, and data augmentation.
- How `deep.py` trains the model, and why the project keeps both a classical and a deep model.

---

## 1. Neurons and layers

A **neuron** is a tiny calculator. It does four things:

1. It takes several input numbers.
2. It multiplies each input by its own **weight** (how much that input matters).
3. It adds them all up and adds one extra number, the **bias**.
4. It passes the total through a simple non-linear function. The most common is **ReLU**: "if the number is negative, output 0; otherwise output it unchanged".

```
 inputs x1, x2, ... xn
        │
        ▼
 ┌─────────────────────────────┐
 │ multiply each by its weight │
 └─────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────┐
 │ add them up, then add bias  │
 └─────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────┐
 │ ReLU: negative becomes 0    │
 └─────────────────────────────┘
        │
        ▼
     output
```

The weights and biases are the **parameters**. At the start they are random (or, as we will see, copied from a previous model). **Training** means adjusting them until the outputs become useful.

A **layer** is many neurons looking at the same inputs. A **neural network** is layers stacked, so one layer's output is the next layer's input.

Why the non-linear step matters: without it, any number of stacked layers collapses into one big multiplication. That could only draw straight-line boundaries. The non-linearity is what lets extra depth add real power.

Intuition for images: early layers detect simple things (edges). Later layers combine them into corners and textures. Later still come blobs, parts and whole structures.

---

## 2. Convolutional neural networks (CNNs)

Imagine feeding a 224x224 image into ordinary layers. Every neuron would connect to all 50,176 pixels. That is wasteful. It also ignores a fact: an edge is an edge wherever it appears in the picture.

A **convolutional layer** is smarter. It slides a small window across the image. The window is called a **filter** or **kernel**, for example a 3x3 grid of numbers.

At each position it does the following:

```
 ┌──────────────────────────────────┐
 │ Place the 3x3 filter on the image│
 └──────────────────────────────────┘
                 │
                 ▼
 ┌──────────────────────────────────┐
 │ Multiply each covered pixel by   │
 │ the filter number above it       │
 └──────────────────────────────────┘
                 │
                 ▼
 ┌──────────────────────────────────┐
 │ Add the 9 products: one number   │
 └──────────────────────────────────┘
                 │
                 ▼
 ┌──────────────────────────────────┐
 │ Slide one step; repeat everywhere│
 │ Result grid = "feature map"      │
 └──────────────────────────────────┘
```

The feature map shows where that filter's pattern appears in the image.

**Link to Lesson 03.** There you met a hand-designed Gaussian blur kernel (used inside unsharp masking). That is exactly the same operation as a CNN layer: slide a small grid of fixed numbers over the image and sum. (The median and bilateral filters also slide a window, but they are non-linear, so they are not convolutions.)

The difference is who picks the numbers:

| | Lesson 03 filters | CNN filters |
|---|---|---|
| Who chose the numbers? | A human, from theory | The training process, from data |
| How many? | A handful, one at a time | Thousands, in many layers |
| Stacking | Usually one step | Filters operate on the feature maps of earlier filters |

So a CNN is "image filtering where the computer discovers the useful filters itself". Early filters usually end up looking like edge and blob detectors anyway. Deeper ones respond to more abstract patterns.

**Pooling.** After convolutions, a network often shrinks the feature maps.

- **Max pooling** keeps only the largest value in each small block (for example 2x2).
- **Global average pooling** averages a whole map down to one number.

This makes the network cheaper and slightly insensitive to the exact pixel position of a pattern. ResNet-18 uses a max-pool near the start and a global average pool at the end, just before its final classifier layer.

---

## 3. Residual connections (the "Res" in ResNet)

Very deep networks are hard to train. The learning signal (the gradient, see section 6) fades as it travels backwards through many layers. Think of a message whispered down a long line of people.

A **residual connection**, also called a skip connection, adds a block's input directly to its output. The signal always has a short path back. The project's code comments say this is what lets gradients reach the early layers being fine-tuned. ResNet-18 has 18 weight layers.

```
 input ─────────────────────┐   (skip: short path)
   │                        │
   ▼                        │
 ┌──────────────┐           │
 │ a few conv   │           │
 │ layers       │           │
 └──────────────┘           │
   │                        │
   ▼                        ▼
 output of layers  ──►  ( + add )  ──►  block output
```

---

## 4. Softmax: scores into probabilities

The last layer outputs one raw score per class. Here there are 4: No Tumor, Glioma, Meningioma, Pituitary. These scores can be any numbers.

**Softmax** converts them into probabilities that are positive and add up to 1. Bigger scores get bigger shares.

```
 raw scores  [ 2.0, 1.0, 0.1, 0.1 ]
      │
      ▼
 ┌─────────────────────────────────────┐
 │ raise e to each score               │
 │ divide each by the sum of them all  │
 └─────────────────────────────────────┘
      │
      ▼
 probabilities  [ 0.61, 0.22, 0.09, 0.09 ]   (sum = 1)
      │
      ▼
 predicted class = the largest (first one)
```

These probability vectors are what the project blends with the classical model.

---

## 5. Loss: measuring how wrong the model is

The **loss** is one number saying how bad the current predictions are. Training tries to make it small.

For classification the standard choice is **cross-entropy**. For each training image, look at the probability the model gave to the *correct* class, then compute `-log(p_correct)`.

| Probability given to the correct class | Loss | Meaning |
|---|---|---|
| 0.9 | 0.105 | small |
| 0.1 | 2.30 | large |
| 0.01 | 4.6 | very large |

So being confidently wrong is punished heavily. Being confidently right costs almost nothing.

---

## 6. Gradient descent and backpropagation

Picture the loss as a hilly landscape. Each position on the map is one setting of all the weights. The height is the loss. We want to walk downhill, like a hiker in fog who can only feel the slope under their feet.

- **Gradient**: for each weight, the slope. It answers "if I increase this weight slightly, does the loss go up or down, and by how much?"
- **Backpropagation**: an efficient way to compute the gradient for *every* weight. It works backwards from the loss through the layers, using the chain rule from calculus.
- **Gradient descent**: nudge each weight a small step in the direction that lowers the loss.
- **Learning rate**: the size of that step. Too large and you overshoot the valley. Too small and training crawls.

Real training uses **mini-batches**. Instead of computing the loss over all images, use a small group (32 here). This gives a noisy but cheap estimate of the slope. It is called **stochastic** gradient descent.

Terms:

- **Batch**: the group of images processed in one update step.
- **Epoch**: one full pass through the whole training set. With 2439 training images and batch size 32, one epoch is about 77 update steps.

The training loop, repeated for every batch:

```
 ┌──────────────────────────────┐
 │ 1. Clear the old gradients   │
 └──────────────────────────────┘
               │
               ▼
 ┌──────────────────────────────┐
 │ 2. Forward pass: feed the    │
 │    batch in, get predictions │
 └──────────────────────────────┘
               │
               ▼
 ┌──────────────────────────────┐
 │ 3. Loss: how wrong are they? │
 └──────────────────────────────┘
               │
               ▼
 ┌──────────────────────────────┐
 │ 4. Backpropagation: compute  │
 │    gradient of every weight  │
 └──────────────────────────────┘
               │
               ▼
 ┌──────────────────────────────┐
 │ 5. Optimiser step: nudge     │
 │    every weight downhill     │
 └──────────────────────────────┘
               │
               ▼
        next batch ...
```

### Three training ingredients in plain words

**AdamW (the optimiser).** An optimiser is the rule that turns gradients into weight changes. Plain gradient descent uses one step size for everything. **Adam** adapts the step size for each weight separately: weights with steady, consistent gradients take confident steps, noisy ones take careful steps. **AdamW** adds *weight decay*: a gentle pull of all weights toward zero each step. Large weights are a sign of an over-complicated model, so this pull discourages overfitting. The project uses weight decay of 1e-4.

**Cosine schedule (changing the learning rate over time).** A fixed step size is not ideal. Big steps are useful early, when we are far from a good solution. Small steps are better late, for fine settling. The cosine schedule starts at the full learning rate and lowers it smoothly along a cosine curve (a gentle downward S shape) until it is almost zero at the end of training. Think of parking a car: fast at first, then slower and slower.

**Class weights (in the loss).** Our data has far fewer "No Tumor" images than tumor images. Without help, the network could ignore the rare class. Class weights multiply each image's loss by a factor that is larger for rarer classes. The weight is computed as: total images divided by (number of classes times images in that class). A rare class therefore gets a bigger say. This is the same idea as the classical models' balanced class weights in Lesson 6.

---

## 7. Overfitting, and how the code fights it

**Overfitting** is memorising the training images, noise included, instead of learning patterns that carry over to new scans. Sign: training accuracy keeps rising while validation accuracy stalls or falls.

The project uses several defences:

| Defence | Plain-words explanation |
|---|---|
| **Dropout (0.4)** | Before the final layer, randomly switch off 40% of the inputs during training. The model cannot lean on any single feature. |
| **Weight decay (1e-4)** | The gentle pull toward zero inside AdamW. |
| **Label smoothing (0.05)** | The target is "95% this class" instead of "100%". |
| **Validation split and best-epoch restore** | After each epoch, score the model on held-out validation images. At the end, restore the weights from the best epoch, not simply the last one. |
| **Data augmentation** | Show varied versions of each image (section 9). |

**Label smoothing in detail.** Normally the correct answer is written as "100% this class, 0% everything else". But our dataset's labels are imperfect (for example, its glioma split is known to be inconsistent). Training the network to be 100% certain about possibly wrong labels teaches it to be overconfident. With smoothing 0.05, the target becomes about 95% for the right class, and the remaining 5% is spread over the other classes. The network stays humble.

---

## 8. Transfer learning

Training a CNN from scratch needs a huge number of images. **Transfer learning** starts from a network that has already been trained on a large dataset, then adapts it to your task. It is like hiring someone who already knows how to read and teaching them a new subject, rather than teaching a baby to read first.

Our ResNet-18 is **pretrained on ImageNet**: about a million everyday photos in 1000 categories (dogs, cars, and so on; no medical images). Its early filters already detect edges, textures and blobs, which are useful for MRI too. We then **fine-tune**: we keep training on our MRI images so the weights adapt.

```
 ┌──────────────────────────────────────┐
 │ ResNet-18 pretrained on ImageNet     │
 │ (trunk + 1000-class final layer)     │
 └──────────────────────────────────────┘
                  │
                  ▼
 ┌──────────────────────────────────────┐
 │ Remove the 1000-class final layer    │
 └──────────────────────────────────────┘
                  │
                  ▼
 ┌──────────────────────────────────────┐
 │ Attach new head:                     │
 │   Dropout (0.4)                      │
 │   then Linear layer with 4 outputs   │
 └──────────────────────────────────────┘
                  │
                  ▼
 ┌──────────────────────────────────────┐
 │ Fine-tune on MRI images:             │
 │   trunk learning rate 3e-4 (gentle)  │
 │   head  learning rate 1e-3 (faster)  │
 │   cosine schedule lowers both        │
 └──────────────────────────────────────┘
```

Why two learning rates? The pretrained trunk is already good, so it should move gently. The brand-new head starts random, so it needs bigger steps to learn quickly.

**Grayscale to colour.** MRI is grayscale, but ImageNet models expect three colour channels. The code copies the grey channel three times. It then normalises with the ImageNet mean and standard deviation, which is what the pretrained weights were trained with.

---

## 9. Data augmentation

Augmentation creates varied versions of each training image on the fly. The model sees a slightly different version every epoch and learns to ignore nuisance differences. It is applied to training images only.

Transforms used:

| Transform | Setting |
|---|---|
| Random crop, resized back | keeps 78-100% of the frame |
| Rotation | plus or minus 15 degrees |
| Horizontal flip | probability 0.5 |
| Contrast change | x0.85 to x1.15 |
| Brightness change | plus or minus 18 |
| Gamma change | probability 0.3 |

Why these specific ones? The training and test halves of the dataset come from visibly different sources, so the model must not rely on image size, framing or brightness quirks. All the transforms keep the label valid: a flipped glioma is still a glioma.

**Test-time augmentation (TTA).** Use the same trick when predicting. Instead of asking the network once, ask it 4 times with slightly different views of the same image, then average the answers. It is like getting a second opinion from four angles.

```
 one test image
      │
      ├──► original,  full frame      ──► probabilities A
      ├──► mirrored,  full frame      ──► probabilities B
      ├──► original,  90% zoom        ──► probabilities C
      └──► mirrored,  90% zoom        ──► probabilities D
                                              │
                                              ▼
                              average A, B, C, D = final probabilities
```

---

## 10. Why both a classical model and a deep model?

The project has two branches (see Lessons 5 and 6):

| | Classical branch | Deep branch |
|---|---|---|
| Input | 171 hand-engineered measurements from the segmented region | The enhanced image itself |
| Strength | Interpretable: you can show the region and numbers | Learns patterns we did not think to measure |
| Result | Excellent for "is there a tumor?" | Needed for "which kind?" |

The measured evidence for 4-class typing:

- Engineered features alone: 69.29% accuracy (glioma recall 23%, pituitary 45.95%).
- Fine-tuned CNN: **80.20% accuracy and 79.48% balanced accuracy**.
- Detection (tumor vs none) is handled well by the classical SVM at 95.43% accuracy and 93.77% sensitivity, and a human can audit it.

The final typing output is a **weighted blend** of the two probability vectors (done by `blend_probabilities`). The weight was chosen on a validation split of 431 images carved from the training set, not the test set. The search chose 0.95 for the CNN.

Honest consequence: on this test set the blend gives the same numbers as the CNN alone. The classical branch remains for the segmentation, measurements and audit trail.

```
 ┌───────────────┐      ┌────────────────┐
 │ Classical     │      │ CNN probs      │
 │ probabilities │      │ (with TTA)     │
 └───────────────┘      └────────────────┘
        │ weight 0.05           │ weight 0.95
        └───────────┬───────────┘
                    ▼
        ┌────────────────────────┐
        │ Weighted average       │
        └────────────────────────┘
                    ▼
            final typing answer
```

---

## Where this lives in our code

Everything below is in `backend/src/braintumor/deep.py`.

| Stage | Where | What happens |
|---|---|---|
| Build model | `build_backbone` | Load ImageNet-pretrained ResNet-18; replace its final layer with Dropout (0.4) then a 4-output linear layer |
| Prepare data | `_MRIDataset` | Grey to 3 channels, ImageNet normalisation, augmentation for training only |
| Loss | `DeepTumorClassifier.fit` | Cross-entropy with class weights and label smoothing 0.05 |
| Optimiser | `DeepTumorClassifier.fit` | AdamW, trunk rate 3e-4, head rate 1e-3, weight decay 1e-4 |
| Schedule | `DeepTumorClassifier.fit` | Cosine annealing over all epochs |
| Pick best | `DeepTumorClassifier.fit` | Keep the epoch with best validation balanced accuracy |
| Predict | `_infer`, `predict_proba` | Softmax; 4-view TTA average |
| Device | `pick_device` | Prefers Apple Metal (MPS), then CUDA, then CPU |

Settings: 14 epochs, batch size 32, input 224x224, validation fraction 0.15, TTA on. Training took 396 seconds on Apple Metal.

The training flow as a whole:

```
 ┌────────────────────────────────────┐
 │ Build ResNet-18 + new 4-class head │
 └────────────────────────────────────┘
                  │
                  ▼
 ┌────────────────────────────────────┐
 │ Set up: weighted, smoothed loss;   │
 │ AdamW; cosine schedule             │
 └────────────────────────────────────┘
                  │
                  ▼
 ┌────────────────────────────────────┐
 │ Repeat for 14 epochs:              │
 │   for each batch of 32 (augmented):│
 │     guess, loss, backprop, nudge   │
 │   score on validation images       │
 │   remember weights if best so far  │
 └────────────────────────────────────┘
                  │
                  ▼
 ┌────────────────────────────────────┐
 │ Restore best-epoch weights         │
 └────────────────────────────────────┘
```

**Tests**: `backend/tests/test_deep.py` checks that the head matches the class count, outputs are valid probability distributions (also with TTA), augmentation really changes images, the best epoch is selected, blending is a proper weighted average, and a saved model reloads with identical predictions. They run on a tiny synthetic dataset and skip if torch is not installed.

---

## Common viva questions

**Q1. Why a CNN instead of a normal neural network for images?**
A convolutional layer reuses one small filter across the whole image. It needs far fewer parameters, and it recognises a pattern wherever it appears. That fits the idea that an edge or texture means the same thing anywhere.

**Q2. How are CNN filters different from the filters in Lesson 03?**
The operation is the same: sliding a small grid of numbers over the image. In Lesson 03 a human chose the numbers (for example the Gaussian kernel). In a CNN the numbers are parameters learned by gradient descent from labelled data. There are many of them, and deeper layers filter the outputs of earlier ones.

**Q3. Did you train the network from scratch?**
No. We fine-tuned a ResNet-18 pretrained on ImageNet (transfer learning). Early layers already detect edges and textures, so a dataset of about 2,900 images is enough. Only the final layer was replaced with a new 4-class head.

**Q4. Explain cross-entropy and softmax.**
Softmax turns the 4 raw scores into probabilities that sum to 1. Cross-entropy is `-log` of the probability given to the true class. It is small when the model is confidently right and large when confidently wrong. Training minimises it.

**Q5. What are backpropagation and gradient descent?**
Backpropagation computes, for every weight, how much the loss would change if that weight changed (the gradient). It applies the chain rule backwards through the layers. Gradient descent then moves each weight a small step (set by the learning rate) in the direction that reduces the loss.

**Q6. Why augmentation, and why those specific transforms?**
We measured that the dataset's train and test halves differ in size, framing and brightness. Crops, rotation, flips and intensity changes teach the model to ignore those differences. All of them preserve the tumor-type label. The same logic motivates test-time augmentation.

**Q7. If the CNN is better at typing, why keep the classical model?**
Detection is already handled well by the interpretable classical model, which also beats the typing model on sensitivity. It provides the segmentation, measurements and an audit trail that a CNN cannot, which matters in a medical setting. The CNN is a "second opinion" for the hard typing question.

**Q8. What do AdamW, the cosine schedule and label smoothing each do?**
AdamW adapts the step size per weight and shrinks weights slightly each step to resist overfitting. The cosine schedule lowers the learning rate smoothly from large to nearly zero. Label smoothing makes the target 95% rather than 100%, so the network is not pushed to be overconfident on imperfect labels.

---

## Check yourself

1. In one sentence, what does a neuron compute?
<details><summary>Answer</summary>A weighted sum of its inputs plus a bias, passed through a non-linear function such as ReLU.</details>

2. Our network predicts probabilities `[0.70, 0.10, 0.15, 0.05]` for an image whose true class is the second one. Is the cross-entropy loss small or large, and why?
<details><summary>Answer</summary>Large. The correct class received only 0.10, so the loss is `-log(0.10) = 2.30`. The loss looks only at the probability given to the true class.</details>

3. With 2439 training images and a batch size of 32, roughly how many weight updates happen per epoch, and what is an epoch?
<details><summary>Answer</summary>About 77 (2439 / 32, rounded up). An epoch is one complete pass through the training set.</details>

4. Why does the code use a smaller learning rate for the trunk than for the head?
<details><summary>Answer</summary>The trunk holds pretrained, already useful features that should only be adjusted gently. The head starts random and must learn quickly. A single large rate could destroy the transferred features.</details>

5. Why is the ensemble weight chosen on a validation split rather than the test set?
<details><summary>Answer</summary>Choosing it on the test set would let the test data influence the model, so the reported test score would be optimistically biased. Validation data keeps the test set an honest, untouched estimate.</details>

6. What does test-time augmentation do, and how many views does the project use?
<details><summary>Answer</summary>It predicts on several slightly altered copies of the same image and averages the probabilities. The project uses 4 views: original and mirrored, each at full frame and 90% zoom.</details>

Next: [Lesson 08 - Evaluation Metrics](08_EVALUATION_METRICS.md)
