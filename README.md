# ECG Classification using Transformers

This repository implements two transformer-based models for classifying electrocardiogram (ECG) signals. The models leverage attention mechanisms to effectively process time-series medical data, demonstrating their applicability in cardiac arrhythmia detection.

- **Binary Classification**: Distinguishes between normal and abnormal heartbeats using the PTB Diagnostic ECG Database (PTBDB).
- **Multiclass Classification**: Categorizes heartbeats into five types using the MIT-BIH Arrhythmia Database.

The project highlights how transformers can outperform traditional methods in handling sequential biomedical signals, with a focus on handling variable-length inputs, class imbalance, and interpretability through attention weights.

---

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Binary Classification (PTB Dataset)](#binary-classification-ptb-dataset)
  - [Dataset](#dataset-ptb)
  - [Model Architecture](#model-architecture-ptb)
  - [Code Structure](#code-structure-ptb)
  - [Training & Results](#training--results-ptb)
  - [Inference](#inference-ptb)
  - [Usage](#usage-ptb)
- [Multiclass Classification (MIT-BIH Dataset)](#multiclass-classification-mit-bih-dataset)
  - [Dataset](#dataset-mit-bih)
  - [Model Architecture](#model-architecture-mit-bih)
  - [Code Structure](#code-structure-mit-bih)
  - [Training & Results](#training--results-mit-bih)
  - [Inference](#inference-mit-bih)
  - [Usage](#usage-mit-bih)
- [How to Run](#how-to-run)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Installation

To set up the project environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ecg-transformer-classification.git
   cd ecg-transformer-classification
   ```

2. Install the required Python packages (Python 3.8+ recommended):
   ```bash
   pip install torch numpy pandas scikit-learn matplotlib tqdm
   ```

   For GPU acceleration, ensure you have PyTorch installed with CUDA support (see [PyTorch installation guide](https://pytorch.org/get-started/locally/)).

---

## Project Structure

The project is organized into two main directories, each containing a Jupyter notebook, dataset files, and related artifacts:

```
ecg-transformer-classification/
├── ECG_ptb_binary_classifier/
│   ├── ECG_ptb_binary_classifier.ipynb  # Notebook for binary classification
│   ├── best_model.pt                    # Saved best model checkpoint
│   ├── ptbdb_normal.csv                 # Normal ECG signals
│   └── ptbdb_abnormal.csv               # Abnormal ECG signals
├── ECG_mitbih_multiclass_classifier/
│   ├── ECG_mitbih_multiclass_classifier.ipynb  # Notebook for multiclass classification
│   ├── mitbih_train.csv                 # Training data
│   └── mitbih_test.csv                  # Test data
├── .git/                                # Git repository
└── README.md                            # This documentation
```


## Binary Classification (PTB Dataset)

### Dataset (PTB)

- **Source**: PTB Diagnostic ECG Database from PhysioNet.
- **Files**: `ptbdb_normal.csv` (normal signals) and `ptbdb_abnormal.csv` (abnormal signals).
- **Classes**: 0 (normal), 1 (abnormal).
- **Preprocessing**: Signals are fixed-length (187 time steps); data is concatenated and split into train (70%), validation (15%), and test (15%) sets with stratification.
- **Total Samples**: ~14,551 (balanced between classes).


### Model Architecture (PTB)

- **Input Shape**: (batch_size, seq_len, 1) where seq_len=187.
- **Embedding**: Linear projection from 1 to 64 dimensions.
- **Positional Encoding**: Sinusoidal (fixed).
- **Transformer Encoder**: 1 layer with 4 attention heads (using `nn.MultiheadAttention`).
- **Pooling**: Masked mean pooling (ignores zero-padded regions).
- **Classifier**: Linear layer to 2 outputs.
- **Total Parameters**: ~20K (lightweight for quick training).

### Code Structure (PTB)

1. **Data Loading**: Read CSVs, assign labels, and split datasets.
2. **Dataset Class**: Custom `ECGDataset` for PyTorch tensors.
3. **DataLoaders**: Batch size=64.
4. **Padding Mask**: Boolean mask for non-zero values.
5. **Positional Encoding**: Precomputed sinusoidal matrix.
6. **Encoder Layer**: Custom wrapper with attention, residuals, and feed-forward.
7. **Model**: `ECGTransformer` integrates all components.
8. **Early Stopping**: Monitors validation loss; saves best model.
9. **Training**: 7 epochs max, with metrics tracking.
10. **Evaluation**: Load best model for test metrics and visualization.

### Training & Results (PTB)

Trained on CPU/GPU with Adam optimizer (lr=1e-3) and CrossEntropyLoss. Early stopping triggered after minimal epochs due to rapid convergence.

- **Best Validation Accuracy**: 100% (saved as `best_model.pt`).
- **Test Metrics**:
  - Accuracy: 1.00
  - Precision (macro): 1.00
  - Recall (macro): 1.00
  - F1-score (macro): 1.00
  - 
<img width="903" height="386" alt="Screenshot 2026-03-09 140346" src="https://github.com/user-attachments/assets/f91de73c-8f56-4697-a34d-19fabdbf2c17" />

Confusion Matrix:
```
[[ 607    0]
 [   0 1576]]
```

### Inference (PTB)

To infer on new data:
```python
import torch
model = ECGTransformer(seq_len=187)  # Recreate model
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
input_tensor = torch.tensor(signal).unsqueeze(0).unsqueeze(-1)  # Shape: (1, 187, 1)
with torch.no_grad():
    logits = model(input_tensor)
predicted_class = torch.argmax(logits, dim=1).item()
```

Example: Random test sample prediction with plot.

<img width="796" height="590" alt="Screenshot 2026-03-09 140259" src="https://github.com/user-attachments/assets/c1384ed2-5800-456b-87ad-d03a9ed90510" />

### Usage (PTB)

Open `ECG_ptb_binary_classifier.ipynb` in Jupyter/Colab. Run cells sequentially to train or load `best_model.pt` for inference.

---

## Multiclass Classification (MIT-BIH Dataset)

### Dataset (MIT-BIH)

- **Source**: MIT-BIH Arrhythmia Database from PhysioNet.
- **Files**: `mitbih_train.csv` (training) and `mitbih_test.csv` (testing).
- **Classes**: 0 (normal), 1 (supraventricular premature), 2 (premature ventricular contraction), 3 (fusion), 4 (unclassifiable).
- **Preprocessing**: Trim trailing zeros (variable lengths); upsample classes 1 and 3 to 2000 samples via time-series interpolation + Gaussian noise to address imbalance.
- **Post-Upsampling Train Distribution**:
  | Class | Samples |
  |-------|---------|
  | 0     | 72471   |
  | 1     | 2000    |
  | 2     | 5788    |
  | 3     | 2000    |
  | 4     | 6431    |

### Model Architecture (MIT-BIH)

- **Input Shape**: Variable-length sequences padded to max in batch.
- **Embedding**: Linear from 1 to 64.
- **Positional Encoding**: Learnable.
- **Transformer Encoder**: 2 layers, each with 4 heads (custom `MultiHeadSelfAttention` from scratch).
- **Pooling**: Masked mean pooling.
- **Classifier**: Linear to 5 outputs.
- **Total Parameters**: ~50K.

### Code Structure (MIT-BIH)

1. **Data Loading**: Read CSVs, trim zeros.
2. **Upsampling**: Custom time-series SMOTE-like method for minorities.
3. **Dataset Class**: `TimeSeriesDataset` for variable lengths.
4. **Collate Function**: Pad batches, create masks.
5. **Attention Module**: From-scratch multi-head self-attention with masking.
6. **Encoder Layer**: Attention + feed-forward + residuals.
7. **Model**: `TimeSeriesTransformer` with learnable positions.
8. **Training**: 30 epochs, loss plotting.
9. **Evaluation**: Test metrics including confusion matrix.

### Training & Results (MIT-BIH)

Trained with Adam (lr=1e-3), CrossEntropyLoss. Loss curve shows steady improvement.

<img width="772" height="612" alt="Screenshot 2026-03-09 134102" src="https://github.com/user-attachments/assets/047d7ace-bf02-47c0-9d2a-78446fe08ddd" />

- **Test Metrics**:
  - Accuracy: 0.9755
  - Precision (macro): 0.9010
  - Recall (macro): 0.8534
  - F1-score (macro): 0.8756

Confusion Matrix:
```
[[17977    69    38    17    17]
 [  152   387    14     0     3]
 [   96     7  1322    18     5]
 [   30     0    19   113     0]
 [   43     0     7     1  1557]]
```

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.98      | 0.99   | 0.99     | 18118   |
| 1     | 0.84      | 0.70   | 0.76     | 556     |
| 2     | 0.94      | 0.91   | 0.93     | 1448    |
| 3     | 0.76      | 0.70   | 0.73     | 162     |
| 4     | 0.98      | 0.97   | 0.98     | 1608    |

<img width="702" height="567" alt="Screenshot 2026-03-09 134040" src="https://github.com/user-attachments/assets/76aaa3b1-3a8f-4c4d-8b05-b695467b764b" />


### Inference (MIT-BIH)

Similar to binary:
```python
model = TimeSeriesTransformer(max_len=187, d_model=64)  # Adjust params
model.load_state_dict(torch.load('final_model.pt'))  # If saved
model.eval()
# Prepare padded input and mask
```

The notebook includes a random test sample plot with prediction.

### Usage (MIT-BIH)

Open `ECG_mitbih_multiclass_classifier.ipynb`. Run to train/evaluate.

---

## How to Run

1. Download datasets from PhysioNet if not included.
2. Launch Jupyter: `jupyter notebook`.
3. Open the desired notebook and execute.

For Colab: Upload notebooks and data.

---

## Contributing

Contributions welcome! Fork the repo, create a branch, and submit a pull request. Suggestions: Add CNN hybrids, more datasets, or attention visualizations.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## References

- PTBDB: [PhysioNet](https://physionet.org/content/ptbdb/1.0.0/)
- MIT-BIH: [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
- Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
