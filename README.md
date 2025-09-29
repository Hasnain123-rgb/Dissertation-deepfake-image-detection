#  Improving Deepfake Model Generalization with Augmentations, Perturbations, and Loss Function Analysis

This project explores how various **image augmentations**, **perturbations**, and training strategies impact the **generalization performance** of deepfake detection models across different datasets.

We use 5 popular **baseline architectures** — including both **CNNs** and **Transformers** — trained on the **FF++ (HQ)** dataset. No architectural modifications are made to the models themselves; instead, we analyze how different data-level interventions affect performance.

Models are evaluated on **CelebDF-v2** and **WildDeepfake**, which represent challenging cross-domain testing conditions.

In addition to classification metrics like **AUC**, **Accuracy**, and **EER**, we use **Grad-CAM visualizations** to understand how attention maps shift depending on augmentation strategy.


---

##  Project Pipeline Overview

The following diagram illustrates the complete deepfake detection pipeline — from input preprocessing and augmentation to model inference and evaluation:

![Model Pipeline Overview](Figures/model_pipeline_overview.png)

**Figure:** *Pipeline stages include:*
- Batch of input face images
- Resizing and normalization
- Application of a selected augmentation strategy
- Feature extraction using one of the baseline models
- Prediction (Real/Fake)
- Evaluation via AUC, ROC curves, confusion matrices, and training curves

---

##  Repository Structure

The repository is organized by model architecture. Each folder contains Jupyter notebooks for training and evaluating deepfake detection models with different augmentation strategies.



Dissertation-deepfake-image-detection/

│
├── ResNet50/

├── ConvNeXt/

├── Vision Transformer (ViT)/

├── EfficientNet/

├── Swin Transformer/

├── Figures/

└── README.md


Each model folder includes multiple `.ipynb` files — one for each augmentation strategy (e.g., Compression, MixMo, Frequency Transform, etc.).

---

---

##  Augmentations Used

This project evaluates the impact of the following **11 image augmentations**, grouped by the type of transformation they apply to the training data:

| Category               | Techniques                                                  |
|------------------------|-------------------------------------------------------------|
| Baseline               | No Augmentation                                             |
| Compression-Based      | JPEG Compression                                            |
| Color-Based            | Color Transform                                             |
| Frequency-Based        | Frequency Transform                                         |
| Occlusion-Based        | Hide & Seek                                                 |
| Patch-Based            | Gridmask, Random Mix                                        |
| Mix-Based              | SnapMix, MixMo (α=1), MixMo (α=2)                           |
| Spatial-Based          | YoCo (You Only Cut Once)                                    |

> Each augmentation was applied independently to evaluate its impact on model accuracy, robustness, and generalization across datasets.

### Augmentation Visuals

The figure below illustrates how different augmentation strategies modify the input face images:

![Augmentation Examples](Figures/augmentation_visuals.png)
