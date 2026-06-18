# Project Overview: LSTM-Autoencoder Feature Selection Comparison

## 1. Introduction
This project implements an **LSTM-Autoencoder** for anomaly detection using the **CSE-CICIDS2018** dataset. The primary goal is to evaluate and compare the impact of three different **Feature Selection** algorithms on the model's ability to detect network intrusions.

The project focuses on creating a memory-efficient pipeline capable of handling over 16 million rows of network traffic data on a machine with limited RAM, while maintaining the temporal integrity of the data required for LSTM models.

---

## 2. Dataset: CSE-CICIDS2018
The dataset consists of multiple CSV files, each representing a day of network traffic. 
- **Total Volume**: ~16 million+ rows and ~80+ columns.
- **Classes**: Benign (Normal) and various Attack types.
- **Target**: The model is trained as an Autoencoder on **Benign** data only, learning to reconstruct normal traffic. Anomalies are detected when the reconstruction error exceeds a certain threshold.

---

## 3. Feature Selection Methodology
To find the optimal subset of features, four different paradigms of feature selection are compared:

| Method | Type | Description |
| :--- | :--- | :--- |
| **Mutual Information (MI)** | Filter | Measures the statistical dependence between each feature and the target label. |
| **Random Forest Importance** | Embedded | Uses the Gini importance/mean decrease in impurity from a Random Forest model. |
| **Recursive Feature Elimination (RFE)** | Wrapper | Recursively removes the least important features using a Random Forest estimator. |
| **mRMR** | Filter | Selects features with high relevance to the target and low redundancy with already selected features. |

### Feature Selection Strategy for Large Data
Because loading 16M+ rows into memory for feature selection is computationally prohibitive, the project uses a **Bounded Streaming Sample**. A balanced Benign/Attack sample is collected incrementally from CSV chunks, so feature selection never requires a full-file or full-dataset load in RAM.

To keep the schema consistent across all days, the pipeline uses only the **80 columns shared by every CSV file**. Extra columns that appear only in `Thuesday-20-02-2018` (such as flow/IP identifier fields) are excluded from feature selection, training, and evaluation.

Before running MI, RF Importance, RFE, and mRMR, the project performs **Correlation Pruning** on the bounded labeled sample. If two features have absolute Pearson correlation above `0.9`, one of them is removed. The drop decision keeps the feature with higher Mutual Information (MI) to the label; if MI is tied, the feature with higher variance is kept.

For the most expensive selector, **RFE**, the project uses a smaller balanced subset than MI/RF. This keeps wrapper-based selection feasible on limited-memory hardware while preserving the full dataset for final training and evaluation.

---

## 4. Data Pipeline & Memory Optimization
Due to the massive size of the dataset, the project implements a **Streaming/Iterative Architecture**:

### 4.1 Memory Efficiency
- **True Chunked Streaming**: To prevent OOM on 16M+ rows, the project uses a PyArrow-based streaming pipeline that reads data in small row-groups from disk, avoiding loading full files into RAM.
- **Disk-Backed Evaluation**: Evaluation metrics (ROC-AUC, PR-AUC) are computed using memory-mapped files (`np.memmap`), allowing the analysis of millions of samples without exhausting system memory.
- **Incremental Thresholding**: The anomaly threshold is calculated as a running maximum during training, reducing memory complexity from $O(N)$ to $O(1)$.
- **Chunked Preprocessing**: Raw CSVs are converted to Parquet using chunked reading and append-only Parquet writes, so intermediate chunks are never accumulated into a full-file DataFrame.
- **Streaming Feature Scaling**: `StandardScaler` is fit incrementally over streamed training batches instead of loading a full training file into memory.
- **Drop-Row Cleaning**: Missing or invalid numeric rows are removed after numeric coercion and `inf/-inf` cleanup. The pipeline does not impute missing traffic values with synthetic replacements.
- **Type Downcasting**: Numeric feature columns are normalized to `float32` before sampling, Parquet writing, and model input, which keeps memory usage predictable and avoids schema drift across chunks.
- **Bounded Feature Selection**: Correlation pruning and MI/RF/mRMR operate on a bounded sampled matrix, while RFE uses an even smaller balanced subset and lighter estimator settings to remain feasible on low-RAM machines.

### 4.2 Temporal Split Logic (No Shuffle)
To preserve the time-series nature of the traffic for the LSTM, a **Sequential/Temporal Split** is applied per file:
- **Order**: Data is processed in its original sequence (no shuffling).
- **Splitting Rules for Benign Data**:
    - **If Total Benign > 1,000,000**: 
        - First 500,000 rows $\rightarrow$ **Train Set**.
        - Remaining rows $\rightarrow$ **Test Set**.
    - **If Total Benign $\le$ 1,000,000**:
        - 50:50 Split (ensuring Test $\ge$ Train).
- **Test Set Composition**: The test set consists of the Benign test portion and **all** Attack data.

---

## 5. Model Architecture
The model is a **Sequence-to-Sequence LSTM-Autoencoder**.

### 5.1 Structure
1.  **Encoder**: 
    - Input: `(batch, time_steps, num_features)`
    - LSTM layer compresses the sequence into a fixed-size hidden state (latent vector).
2.  **Latent Space**: A compressed representation of the normal traffic pattern.
3.  **Decoder**:
    - Latent vector is repeated `time_steps` times.
    - LSTM layer reconstructs the sequence from the latent representation.
    - **TimeDistributed Linear Layer**: Maps the decoder output back to the original `num_features`.

### 5.2 Hyperparameters
- **Time Steps (Window Length)**: 10
- **Hidden Dimension**: 16
- **Learning Rate**: 0.001
- **Dropout**: 0.2
- **Batch Size**: 64
- **Epochs**: 30
- **Selected Features per Method**: 15
- **Loss Function**: Mean Absolute Error (`nn.L1Loss`)
- **Optimizer**: Adam

---

## 6. Training & Evaluation Process

### 6.1 Training Phase
- **Data**: Only $\text{Benign}_{\text{train}}$ is used.
- **Process**: The model is updated iteratively using data from each file.
- **Thresholding**: After training, the **MAE reconstruction error** for the training set is calculated. The anomaly threshold is set as the **90th percentile of the training MAE distribution**:
  $$\text{Threshold} = P_{90}(\text{Train MAE Reconstruction Errors})$$
- **Artifact Persistence**: Each feature-selection method saves its trained model, fitted scaler, and metadata under `artifacts/<method>/` for reuse.

### 6.2 Evaluation Phase
- **Data**: $\text{Benign}_{\text{test}} + \text{All Attacks}$.
- **Detection**: If the test **MAE reconstruction error** exceeds the threshold, the sample is flagged as an **Anomaly**.
- **Metrics**:
    - Accuracy
    - Precision, Recall, F1-Score (Normal vs Attack)
    - ROC-AUC (Area Under the Receiver Operating Characteristic Curve)
    - PR-AUC (Area Under the Precision-Recall Curve)

---

## 7. Summary of Workflow
1.  **Bounded Streaming Sample** $\rightarrow$ **Correlation Pruning** $\rightarrow$ **Feature Selection** (MI vs RF vs RFE vs mRMR) $\rightarrow$ **4 Feature Sets**.
2.  For each Feature Set:
    - **Iterative Load** $\rightarrow$ **Temporal Split** $\rightarrow$ **Train LSTM-AE on Benign**.
    - **Calculate Train MAE Distribution** $\rightarrow$ **Set P90 Threshold**.
    - **Save Model + Scaler + Metadata**.
    - **Iterative Load** $\rightarrow$ **Test on Benign/Attack** $\rightarrow$ **Compute Metrics**.
3.  **Compare** results to determine the best feature selection method.
