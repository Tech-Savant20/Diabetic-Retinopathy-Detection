# Diabetic Retinopathy Classification using Deep Learning

## Overview

This project classifies the severity of diabetic retinopathy (DR) from retinal fundus photographs into five grades, using transfer learning with EfficientNet-B0. It was built as the project for the L&T EduTech *Certificate in AI & Edge Computing for Industry Applications* (VIT Summer Industrial Internship, 2025).

On the validation set the model reaches **76.47% accuracy** with **Cohen's kappa of 0.628**. It separates healthy from diseased eyes well (No_DR F1-score 0.97), but it cannot yet grade the severe and proliferative stages, so it is not suitable for clinical use without further work.

## Files

| File | Contents |
|------|----------|
| `diabetic_retinopathy_classification (1).ipynb` | Complete Google Colab notebook: data download, training, evaluation (with saved outputs) |
| `Diabetic Retinopathy Classification Report new.pdf` | Project report as submitted on 6 July 2025 |
| `train.csv` | Image IDs and severity labels (0–4) for all 3,662 images |

The figures in this README are taken from the notebook's saved outputs.

## Dataset

[Diabetic Retinopathy 224x224 Gaussian Filtered](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered) (Kaggle), derived from the [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) dataset. Images are already resized to 224×224 and Gaussian-filtered.

| Class | Images | Share |
|-------|--------|-------|
| No_DR | 1,805 | 49.3% |
| Mild | 370 | 10.1% |
| Moderate | 999 | 27.3% |
| Severe | 193 | 5.3% |
| Proliferate_DR | 295 | 8.1% |
| **Total** | **3,662** | **100%** |

The data is split 80/20 with Keras `validation_split`: **2,931 training** and **731 validation** images.

## Preprocessing

- Pixel values rescaled to [0, 1]
- Training-only augmentation: rotation ±15°, width/height shift 10%, shear 0.1, zoom 10%, horizontal flip
- Validation images are only rescaled

## Model

```
Input (224×224×3)
↓
EfficientNet-B0 (ImageNet weights, first 100 layers frozen)
↓
GlobalAveragePooling2D
↓
Dropout(0.3)
↓
Dense(128, ReLU)
↓
Dropout(0.5)
↓
Dense(5, Softmax)
```

- **Parameters:** 4,214,184 total (4,004,961 trainable, 209,223 non-trainable)
- **Optimizer / loss:** Adam, categorical cross-entropy
- **Batch size:** 32
- **Learning rate schedule:** 1e-3 (epochs 1–10), 5e-4 (11–20), 1e-4 (21–30), 5e-5 afterwards
- **Early stopping:** on validation loss, patience 10, best weights restored
- **Epochs:** up to 50; training stopped after 46

## Results (validation set, 731 images)

| Metric | Value |
|--------|-------|
| Accuracy | 0.7647 |
| Weighted precision | 0.6966 |
| Weighted recall | 0.7647 |
| Weighted F1-score | 0.7208 |
| Macro F1-score | 0.45 |
| Cohen's kappa | 0.6279 |
| Quadratic weighted kappa (severity order) | 0.758 |

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| No_DR | 0.95 | 0.98 | 0.97 | 361 |
| Mild | 0.55 | 0.42 | 0.48 | 74 |
| Moderate | 0.59 | 0.87 | 0.70 | 199 |
| Severe | 0.17 | 0.05 | 0.08 | 38 |
| Proliferate_DR | 0.00 | 0.00 | 0.00 | 59 |

Confusion matrix (rows = true class, columns = predicted class):

| | Mild | Moderate | No_DR | Proliferate_DR | Severe |
|---|---|---|---|---|---|
| **Mild** | 31 | 37 | 4 | 0 | 2 |
| **Moderate** | 10 | 173 | 10 | 0 | 6 |
| **No_DR** | 4 | 4 | 353 | 0 | 0 |
| **Proliferate_DR** | 8 | 46 | 3 | 0 | 2 |
| **Severe** | 3 | 33 | 0 | 0 | 2 |

**Note on quadratic weighted kappa.** The notebook prints a QWK of 0.2524. That value is incorrect: Keras assigns class indices alphabetically (Mild=0, Moderate=1, No_DR=2, Proliferate_DR=3, Severe=4), so the quadratic weights did not follow disease severity. Recomputing QWK from the same confusion matrix with the classes in severity order (No_DR, Mild, Moderate, Severe, Proliferate_DR) gives **0.758**.

### What the results mean

- **Screening works well.** Treated as "any DR vs no DR", the model has 95.4% sensitivity (353/370) and 97.8% specificity (353/361). For referable DR (moderate or worse) it has 88.5% sensitivity and 90.1% specificity.
- **Grading the late stages fails.** 0 of 59 proliferative and 2 of 38 severe cases are graded correctly; 46 and 33 of them respectively are predicted as moderate. These patients would still be referred, but the model cannot say how urgent the referral is.
- **Mild vs moderate is confused.** 37 of 74 mild cases are predicted as moderate.

## Limitations

- **Class imbalance:** No_DR outnumbers Severe by about 9 to 1, and the rarest grades are the ones the model misses.
- **Optimistic evaluation:** the same validation set is used both for early stopping (best-weight selection) and for the reported metrics. A separate held-out test set is needed for an unbiased estimate.
- **Resolution:** 224×224 inputs lose small lesions such as microaneurysms.
- **Unstable validation curves:** validation accuracy fluctuates strongly between epochs.

## Possible improvements

- Class weights, oversampling of Severe/Proliferate_DR, or focal loss
- A separate test split, or stratified k-fold cross-validation
- Higher input resolution and an ordinal-aware loss
- Grad-CAM visualisation to check which regions drive predictions
- 8-bit quantisation (about 16 MB → about 4 MB) for deployment on edge devices

## How to run

1. Open `diabetic_retinopathy_classification (1).ipynb` in [Google Colab](https://colab.research.google.com/) and select a GPU runtime.
2. Run the cells in order. The first cell downloads the dataset with `kagglehub` (a Kaggle account may be required).
3. Training takes roughly 45 seconds per epoch on a Colab GPU. The notebook saves the trained model as `diabetic_retinopathy_model.h5` and the metrics as `diabetic_retinopathy_evaluation_results.csv`.

Main libraries: TensorFlow/Keras, NumPy, pandas, scikit-learn, matplotlib, seaborn, kagglehub.

## Screenshot

![Colab notebook run, 6 July 2025](https://github.com/user-attachments/assets/398968ce-ce3f-4613-add9-ddaf72ee0c61)

## References

- M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML, 2019.
- V. Gulshan et al., "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs", JAMA, 316(22), 2016.
- APTOS 2019 Blindness Detection, Kaggle, 2019.

## Author

Abhyuday Tomar (23BCE11727), VIT Bhopal University — [github.com/Tech-Savant20/Diabetic-Retinopathy-Detection](https://github.com/Tech-Savant20/Diabetic-Retinopathy-Detection)
