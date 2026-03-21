# Advanced Customizations in Deep Learning and Neural Networks

**Course:** Deep Learning  
**Author:** Venkata Yashwanth Paladugu
 
**Frameworks:** TensorFlow / Keras and PyTorch

---

## Video Walkthroughs

Each colab has a dedicated video walking through the code line by line.

| Colab | Video Link |
|-------|-----------|
| Part 1 — Regularization and Augmentation (TF/Keras) | [Watch on Google Drive](https://drive.google.com/file/d/1O9w4J9hpEInVUKjga77_npPD7gGHbACx/view?usp=sharing) |
| Part 1k Extended — Video and Document Augmentation | [Watch on Google Drive](https://drive.google.com/file/d/1fHKoxICszKgLAt-NfAsvkOug13BNiNfJ/view?usp=sharing) |
| Part 2 — Advanced Custom Constructs (TF/Keras) | [Watch on Google Drive](https://drive.google.com/file/d/1FhuJWhCGMNiNFI7lHblvDFM74Iio1zec/view?usp=sharing) |
| Part 2 PyTorch — Advanced Custom Constructs (PyTorch) | [Watch on Google Drive](https://drive.google.com/file/d/1LPchyqVvlkUi2M0cBcAffyJsk5vF5Hpt/view?usp=sharing) |

---

## Repository Structure

```
.
├── README.md
├── part1.ipynb
├── part1k_video_document_augmentation.ipynb
├── part2.ipynb
└── part2_pytorch.ipynb
```

---

## Colab 1 — Part 1: Regularization, Augmentation and Generalization (TF/Keras)

**File:** `part1.ipynb`  
**Datasets:** Fashion-MNIST, CIFAR-10, IMDB Reviews, Mauna Loa CO2, Sunspot Activity, Wine Quality, Breast Cancer Wisconsin, TF Speech Commands (Source-Filter Model)

This notebook covers all items in Part 1 of the assignment using TensorFlow and Keras.
Every technique is demonstrated with an A/B test comparing the baseline against the augmented or regularized version.

### Sections

**1a — L1 / L2 Regularization**  
A/B test comparing no regularization against L2 (ridge) and L1 (lasso). L2 adds a squared weight penalty that keeps weights small and smooth. L1 adds an absolute value penalty that drives many weights exactly to zero, useful for feature selection. Both are passed as `kernel_regularizer` arguments to Dense layers. Validation loss and overfit gap are compared across all three.

**1b — Dropout**  
A/B test comparing dropout rates of 0%, 30%, and 50%. Dropout randomly zeros neurons during each training step, preventing co-adaptation and acting like an ensemble of thinned networks. Model names must use only alphanumeric characters because percent signs in names cause TensorFlow scope errors. `clear_session()` is called inside the loop before each model to prevent variable name collisions.

**1c — Early Stopping**  
`callbacks.EarlyStopping` monitors validation loss with patience 5 and restores the best weights automatically. Training is capped at 100 epochs but stops well before that. The plot marks both the best epoch and the actual stop epoch, showing the patience window.

**1d — Monte Carlo Dropout**  
At inference time, dropout is kept active by calling `model(X, training=True)`. Running 100 such forward passes and stacking the results gives a mean prediction and a standard deviation per class. The standard deviation is a Bayesian approximation of the model's uncertainty. High uncertainty indicates ambiguous or out-of-distribution samples. Results are shown as bar charts with error bars per class.

**1e — Weight Initializations**  
Glorot uniform paired with tanh, He normal paired with relu, LeCun normal paired with selu, and random normal paired with relu. Each initializer is mathematically derived for its paired activation. Glorot uses fan-in plus fan-out in the denominator, He uses only fan-in to account for relu zeroing half its inputs, and LeCun uses fan-in for SELU self-normalization. Validation accuracy curves are compared.

**1f — Batch Normalization**  
A/B test at a high learning rate of 0.01. Without BatchNorm, training is unstable at this rate. With BatchNorm placed between the Dense layer and the activation, training is smooth and converges faster. BatchNorm normalizes each mini-batch to zero mean and unit variance, then applies learned gamma and beta parameters.

**1g — Custom Dropout and Custom Regularizer**  
`MCAlphaDropout` subclasses `keras.layers.Layer` and always applies dropout regardless of the training flag, enabling Monte Carlo uncertainty estimation at inference. `MyL1L2Regularizer` subclasses `keras.regularizers.Regularizer` and combines L1 and L2 penalties with separate configurable factors. Both implement `get_config()` for serialization.

**1h — Callbacks and TensorBoard**  
Five callbacks used together: `EarlyStopping`, `ModelCheckpoint` (saves best weights to disk), `ReduceLROnPlateau` (halves LR when val loss stalls for 3 epochs), `TensorBoard` (logs histograms and graph), and a custom `LRLogger` class that prints the learning rate after each epoch. TensorBoard is launched inline.

**1i — Keras Tuner**  
Hyperband search over number of layers (1-2), units per layer (64/128/256), BatchNorm on or off, dropout rate (0.1-0.4), and learning rate (1e-3 or 1e-2). Hyperband discards poor configurations early and allocates more epochs to promising ones. Runtime is approximately 2-3 minutes on GPU with `max_epochs=5`.

**1j — Image Augmentation Pipeline**  
Uses `keras.layers` preprocessing layers — RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomTranslation — baked directly into the model graph. These layers are active only during training and disabled automatically at inference. An A/B test compares a CNN without augmentation against one with augmentation on CIFAR-10. KerasCV's RandomRotation was not used because it relies on `ImageProjectiveTransformV3` which is not registered for XLA GPU in Colab.

**1k-i — Text Augmentation (IMDB Reviews)**  
Real IMDB movie reviews loaded from `keras.datasets.imdb` and decoded using the reversed word index. Six augmentation techniques applied using nlpaug: word deletion, word swap, word split, keyboard typos, random character insertion, and synonym replacement via WordNet.

**1k-ii — Time-Series Augmentation (CO2 and Sunspots)**  
Mauna Loa CO2 dataset (2,284 weekly readings, 1958-2001) and Sunspot Activity dataset (309 annual readings, 1700-2008) loaded from statsmodels. Five augmentation functions: jitter (Gaussian noise scaled to signal std), scaling (random amplitude factor), time warp (smooth temporal distortion via cubic spline), magnitude warp (smooth amplitude modulation via cubic spline), and window slice (random contiguous crop resampled to original length).

**1k-iii — Tabular Augmentation (Wine Quality and Breast Cancer)**  
Wine Quality (178 samples, 13 features, 3 classes) and Breast Cancer Wisconsin (569 samples, 30 features, binary) loaded from sklearn. Techniques: simplified SMOTE (interpolation between minority-class pairs), Gaussian noise scaled per-feature by standard deviation, and feature dropout (replace values with column mean to simulate missing data). Class imbalance in Breast Cancer is corrected using SMOTE.

**1k-iv — Image Augmentation (tf.image)**  
Standard `tf.image` operations applied to CIFAR-10 images: random flip, brightness, contrast, saturation, hue, and crop-then-pad. Applied to the same source image eight times to show the diversity of random transformations.

**1k-v — Audio Augmentation (Source-Filter Speech Model)**  
Speech generated using the acoustic source-filter model of speech production: a glottal pulse train at F0=120 Hz shaped by three bandpass filters at formant frequencies F1=800 Hz, F2=1200 Hz, F3=2500 Hz. This produces a genuine voiced vowel waveform. Six augmentations applied: white noise at two SNR levels, time shift, speed change via resampling, pitch shift, polarity inversion, and gain.

---

## Colab 2 — Part 1k Extended: Video and Document Image Augmentation

**File:** `part1k_video_document_augmentation.ipynb`  
**Purpose:** Completes the Part 1k requirement by covering the two remaining modalities: video and document images.

### 1k-vi — Video Augmentation

A video tensor has shape (T, H, W, C) — T frames each of size H x W with C channels. A synthetic 16-frame 64x64 RGB video is created with a moving yellow circle and a color gradient that shifts over time, giving temporal structure for the augmentations to act on. In Colab with internet access, the UCF-101 dataset can be loaded directly via `tensorflow_datasets`.

The critical design principle: spatial transforms must be applied consistently across all frames. Applying different flips or crops to different frames destroys the motion patterns the model is supposed to learn.

Eight augmentation functions:

- **Temporal Crop:** randomly selects a start frame and returns n consecutive frames. The start is chosen once and applied to the whole clip.
- **Horizontal Flip:** reverses the width axis across all frames with `v[:, :, ::-1, :]`. One operation flips every frame identically.
- **Brightness Jitter:** draws one scalar shift and adds it to the entire video array. Same shift for all frames to avoid flickering.
- **Temporal Reverse:** reverses frame order with `v[::-1].copy()`. Useful for symmetric actions.
- **Frame Drop:** replaces randomly selected frames with the previous frame, simulating packet loss or variable frame rate.
- **Spatial Crop:** computes one crop coordinate pair and applies the same crop to every frame, then resizes each frame back to original dimensions using `cv2.resize`.
- **Gaussian Noise:** adds independent noise per frame, simulating sensor noise in low-light conditions.
- **Color Jitter:** applies a uniform hue shift to all frames simultaneously.

A temporal consistency visualization shows all 16 frames of the horizontal flip side by side, confirming the motion pattern is preserved but mirrored.

### 1k-vii — Document Image Augmentation

Document AI tasks such as OCR, form understanding, and invoice parsing require models robust to scanning artifacts and physical document conditions. A synthetic document is created programmatically with a title bar, paragraph text lines, a table structure, and page margins.

Seven augmentation functions:

- **Perspective Warp:** displaces the four image corners by random amounts and applies a perspective transformation using `cv2.getPerspectiveTransform`. The destination array must be explicitly cast to float32 after `np.clip` because NumPy promotes the dtype to float64, which OpenCV rejects with an assertion error.
- **Gaussian Blur:** applies Pillow's GaussianBlur filter with a random radius, simulating out-of-focus scanning.
- **Brightness and Contrast:** chains Pillow's ImageEnhance.Brightness and ImageEnhance.Contrast with random factors, simulating uneven illumination or faded documents.
- **Salt and Pepper Noise:** sets random pixels to 255 (salt) and 0 (pepper), simulating scanner dust and toner specks.
- **Rotation:** applies a small random rotation up to 3 degrees using Pillow with BICUBIC resampling. Kept small because larger rotations are unrealistic for scanned documents.
- **JPEG Compression:** saves the image to an in-memory BytesIO buffer as JPEG at low quality, then reads it back. This round-trip introduces authentic blocking artifacts around text edges.
- **Shadow Gradient:** multiplies the image by a linear gradient from 1.0 on the left to (1.0 - intensity) on the right, simulating the shadow from a book spine.

A stochastic pipeline applies each transform with probability 0.5, producing a different random combination each call. Five pipeline outputs of the same document are shown side by side.

---

## Colab 3 — Part 2: Advanced Custom Constructs (TF/Keras)

**File:** `part2.ipynb`  
**Datasets:** Fashion-MNIST (classification), California Housing (regression)

This notebook implements all twelve items from Part 2A using TensorFlow and Keras.

### Important: Keras 3.x Compatibility Fixes

Several breaking changes in Keras 3.x required fixes throughout this notebook.

- `metrics=['accuracy']` triggers a dtype promotion bug. Every `model.compile()` uses metric objects: `metrics=[keras.metrics.SparseCategoricalAccuracy()]`
- `keras.backend.set_value()` was removed. Learning rate is set with `optimizer.learning_rate.assign(lr)`
- `add_weight()` requires all keyword arguments: `add_weight(shape=..., initializer=..., name=...)`
- `kernel_constraint` requires a `keras.constraints.Constraint` subclass. Plain functions are not accepted.

### Sections

**i — OneCycleScheduler**  
Subclasses `callbacks.Callback` and overrides `on_epoch_begin`. Phase 1 is linear warmup over `warmup_epochs` epochs. Phase 2 is cosine annealing for the remaining epochs. An A/B test compares OneCycle against constant LR using the same SGD with momentum 0.9 optimizer.

**ii — MCAlphaDropout**  
Subclasses `keras.layers.AlphaDropout` and overrides `call` with a single line that forces `training=True`. AlphaDropout is used instead of standard Dropout because it preserves the mean and variance of SELU activations, maintaining the self-normalizing property. Always-active dropout enables Monte Carlo uncertainty estimation.

**iii — MaxNormDense**  
Custom Dense layer that clips each column of the kernel so its L2 norm does not exceed `max_norm`. The clipping is computed in `call` using `tf.norm` and `tf.minimum`. Weights are accessed after training via `model.get_layer('name').kernel.numpy()` rather than `model.layers[index]` because Activation layers between Dense layers shift the index.

**iv — TensorBoard**  
Uses `callbacks.TensorBoard` with `histogram_freq=1` and `write_graph=True`. A custom `WeightNormLogger` callback uses `tf.summary.create_file_writer` to log the total L2 weight norm and current learning rate as custom scalars each epoch.

**v — HuberLoss**  
Subclasses `keras.losses.Loss`. The call method computes the quadratic component as `tf.minimum(abs_error, delta)` and the linear remainder separately. The model is saved with `model.save()` and reloaded with `keras.models.load_model(..., custom_objects={'HuberLoss': HuberLoss})`. Compared against MSE on California Housing.

**vi — Custom Activation, Initializer, Regularizer, Constraint**  
Four components assembled in one model. `leaky_relu` is a plain function using `tf.maximum`. `MyGlorotInitializer` subclasses `keras.initializers.Initializer` and implements the Glorot formula. `MyL1Regularizer` subclasses `keras.regularizers.Regularizer`. `MyPositiveWeights` subclasses `keras.constraints.Constraint` and clips weights to non-negative values.

**vii — HuberMetric**  
Subclasses `keras.metrics.Mean` which handles the running average. Only `update_state` is overridden to compute the Huber value per batch and pass it to the parent's accumulator via `super().update_state()`.

**viii — Custom Layers**  
Three layers. `AddGaussianNoise` adds noise only when `training=True`. `MyDense` reimplements the Dense layer using `add_weight` with all keyword arguments. `MyLayerNorm` implements Layer Normalization using `tf.nn.moments` and verifies its output matches `keras.layers.LayerNormalization` on the same input.

**ix — ResidualBlock and ResidualRegressor**  
`ResidualBlock` subclasses `keras.layers.Layer`. The projection Dense is created lazily in `build` when the input and output dimensions differ. `ResidualRegressor` subclasses `keras.Model` and stores three ResidualBlocks, training on California Housing.

**x — MyMomentumOptimizer**  
Subclasses `keras.optimizers.Optimizer`. `build` creates velocity accumulators using `add_variable_from_reference`. `update_step` iterates `self._trainable_variables` using `is` comparison to find the matching variable and its velocity, then applies the momentum update rule: v = beta times v plus gradient, w = w minus lr times v.

**xi — Custom Training Loop**  
Uses `tf.GradientTape` to record operations, `tape.gradient` to compute gradients, `tf.clip_by_global_norm` for gradient clipping, and `optimizer.apply_gradients` to apply updates. The `@tf.function` decorator compiles `train_step` and `val_step` to TensorFlow graphs for performance. Streaming metrics accumulate across batches and reset at each epoch start.

**xii — Weights and Biases**  
Uses `WandbMetricsLogger` callback to auto-log all Keras metrics each epoch. Runs in `mode='offline'` for demonstration. A Bayesian sweep configuration is provided with learning rate, dropout, hidden units, and optimizer as search parameters.

---

## Colab 4 — Part 2 PyTorch: Advanced Custom Constructs (PyTorch)

**File:** `part2_pytorch.ipynb`  
**Datasets:** Fashion-MNIST (classification), California Housing (regression)

This notebook mirrors every item in Colab 3 using PyTorch. The same concepts apply with a different API.

### TF/Keras to PyTorch Mapping

| Concept | TF/Keras | PyTorch |
|---------|----------|---------|
| Layer building block | `keras.layers.Layer` | `nn.Module` |
| Trainable model | `keras.Model` | `nn.Module` |
| Custom loss | `keras.losses.Loss` | `nn.Module` with `forward` |
| Custom metric | `keras.metrics.Mean` | Custom Python class |
| Custom optimizer | `keras.optimizers.Optimizer` | `optim.Optimizer` |
| Trainable parameter | `add_weight()` | `nn.Parameter` |
| LR assignment | `optimizer.learning_rate.assign(lr)` | `optimizer.param_groups[0]['lr'] = lr` |
| Dropout at inference | `model(X, training=True)` | `model.train()` before inference |
| Gradient computation | `tf.GradientTape` | `loss.backward()` |
| Apply gradients | `optimizer.apply_gradients()` | `optimizer.step()` |

### Sections

**i — OneCycleLR**  
Uses `optim.lr_scheduler.OneCycleLR` which is built into PyTorch. Called with `scheduler.step()` after each batch, not each epoch. `pct_start=0.3` allocates 30% of steps to warmup.

**ii — MC Dropout**  
`model.train()` is called before inference to keep dropout active. `torch.no_grad()` is still used inside the prediction loop to avoid computing gradients. 100 forward passes are stacked into a (100, batch, 10) tensor.

**iii — MaxNormLinear**  
Wraps `nn.Linear`. Inside `forward`, a `torch.no_grad()` block clips the row norms of `self.linear.weight.data` before computing the linear output. Row norms are used because `nn.Linear` stores weights as (out_features, in_features).

**iv — TensorBoard**  
Uses `torch.utils.tensorboard.SummaryWriter`. `writer.add_graph(model, dummy_input)` logs the computation graph. Tags with forward slashes group related metrics in the TensorBoard UI. `writer.close()` must be called to flush all pending writes.

**v — HuberLossPT**  
Subclasses `nn.Module` and implements `forward`. Logic is identical to the Keras version using PyTorch tensor operations.

**vi — Custom Components**  
`leaky_relu_pt` uses `torch.where`. `glorot_init` is applied with `model.apply(fn)` which recursively visits submodules. `l1_penalty` is added to the loss manually inside the training loop. `apply_max_norm` is called after `optimizer.step()` as a post-step constraint.

**vii — HuberMetricPT and MAEMetricPT**  
Plain Python classes with `reset`, `update`, and `result` methods. `update` uses `.detach()` to prevent gradient tracking through the metric computation.

**viii — Custom Layers**  
`AddGaussianNoisePT` checks `self.training` which PyTorch sets automatically via `model.train()` and `model.eval()`. `MyLinear` stores weights as `nn.Parameter` using He normal initialization. `MyLayerNormPT` uses `x.var(dim=-1, keepdim=True, unbiased=False)` — `unbiased=False` uses the population variance consistent with LayerNorm.

**ix — ResBlockPT and ResRegressorPT**  
`nn.Identity()` handles the pass-through case when input and output dimensions match. Blocks are stored in `nn.ModuleList` — a regular Python list would not register the submodules and their parameters would not appear in `model.parameters()`.

**x — SGDMomentumPT**  
Subclasses `optim.Optimizer`. Velocity state is stored in `self.state[p]`, a per-parameter dict initialized on the first step. In-place operations use the underscore convention: `v.mul_(m).add_(grad)` and `p.data.sub_(v, alpha=lr)`.

**xi — Custom Training Loop**  
The six-step batch loop: `zero_grad`, forward pass, loss computation, `loss.backward()`, `clip_grad_norm_`, `optimizer.step()`. `model.train()` and `model.eval()` switch BatchNorm and Dropout behavior. `torch.no_grad()` wraps the validation loop. `ReduceLROnPlateau` is used without the `verbose` argument which was removed in newer PyTorch versions.

**xii — W&B with PyTorch**  
`wandb.watch(model, log='all', log_freq=50)` hooks into the backward pass to log gradients and parameters automatically. Metrics are logged manually with `wandb.log()` inside the epoch loop.

---

## Running the Notebooks

Open each notebook in Google Colab. Set the runtime to GPU via Runtime > Change runtime type > T4 GPU. Run all cells from top to bottom.

Each notebook installs its own dependencies in Cell 0. No local setup is required.

### Dependencies

```
keras==3.12.0
h5py>=3.11.0,<3.15.0
tensorflow>=2.13
torch
torchvision
keras-tuner
nlpaug
nltk
wandb
statsmodels
scikit-learn
scipy
opencv-python-headless
pillow
matplotlib
numpy
```

---

## References

- Geron, A. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd ed. Chapters 10, 11, 12.
- Smith, L. A Disciplined Approach to Neural Network Hyper-Parameters (2018). 1Cycle policy.
- He et al. Deep Residual Learning for Image Recognition (2016). ResNets.
- Gal and Ghahramani. Dropout as a Bayesian Approximation (2016). MC Dropout.
- Ioffe and Szegedy. Batch Normalization (2015).
- TensorFlow Data Augmentation Tutorial: https://www.tensorflow.org/tutorials/images/data_augmentation
- nlpaug: https://github.com/makcedward/nlpaug
- Weights and Biases: https://wandb.ai
