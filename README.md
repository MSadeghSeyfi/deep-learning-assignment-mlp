## NumPy MLP for Bank Customer Churn (Course Assignment)

This repository contains an educational implementation of a **three-layer MLP built from scratch with NumPy** to predict bank customer churn.
All experiments were executed in **Google Colab**, and the accompanying notebook is fully reproducible.

---

## Features

* End-to-end workflow in a single Colab notebook:

  * Synthetic **bank churn dataset** creation.
  * **Preprocessing**: one-hot encoding, feature scaling, train/val/test split.
  * **MLP implementation from scratch (NumPy)**:

    * Forward/backward propagation,
    * Sigmoid, ReLU, and Tanh activations,
    * Binary cross-entropy loss.
* **Experiments**:

  * Learning rates: 0.1, 0.01, 0.001
  * Hidden sizes: 16, 64, 128
  * L2 regularization & dropout
  * Activation comparison
* **Evaluation**:

  * Accuracy, Recall, F1-score, ROC-AUC
  * Loss & accuracy curves
  * Confusion matrix and ROC curve
* **Feature Importance**:

  * Permutation importance
  * Weight-based inspection

---

## Repository Structure

```
MLP_Bank_Churn.ipynb   # Full solution (run in Google Colab)
requirements.txt        # Minimal dependencies
results/                # Plots & metrics (if saved)
dataset/                # Generated synthetic data (optional)
```

---

## Running the Notebook (Google Colab)

1. Open the notebook in Colab.
2. Install dependencies (if needed):

```python
!pip install numpy pandas scikit-learn matplotlib seaborn
```

3. Run all cells in order to reproduce:

   * Dataset creation
   * Model training
   * Hyperparameter experiments
   * Final evaluation & plots

No GPU is required; the NumPy-based MLP runs on CPU.

---

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
```