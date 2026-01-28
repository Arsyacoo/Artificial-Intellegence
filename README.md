# 🚀 ETH-Sentinel: Hybrid LSTM for Ethereum Price Prediction

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

## 📌 Project Overview
**ETH-Sentinel** is an advanced Deep Learning project designed to predict Ethereum (ETH) price movements using a **Hybrid Feature Approach**. Unlike traditional models that rely solely on historical price data (Technical Analysis), this project integrates **Market Sentiment** (derived from professional news analysis) and **Regime Volatility** to achieve superior Directional Accuracy.

The model is built using a **Pyramid LSTM Architecture** and demonstrates robust performance in capturing non-linear market patterns.

## 🎯 Key Capabilities
*   **Directional Accuracy:** > 80% (Successfully predicts trend Upturns/Downturns).
*   **Hybrid Input:** Combines OHLCV (Hard Data) with Loughran-McDonald Sentiment Scores (Soft Data).
*   **Regime Awareness:** Incorporates "Rolling Volatility" to detect market stability/instability.
*   **Zero-Lag Defense:** utilizes event-based data handling (no cheating/forward-fill) for missing sentiment values.

## 🧠 Model Architecture
The core "Brain" of the system relies on a Stacked LSTM network:
1.  **Input Layer:** 30-Step Lookback Window (Multi-feature).
2.  **LSTM Layer 1:** 128 Units (High-dimensional feature extraction).
3.  **Dropout:** 0.2 (Regularization to prevent Overfitting).
4.  **LSTM Layer 2:** 64 Units (Feature distillation/Pyramid structure).
5.  **Output Layer:** Single-point prediction with Calibration Logic.

## 📊 Performance Metrics (Test Set)
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Directional Accuracy (DA)** | **~87.70%** | Primary success metric. |
| **R2 Score** | **0.98** | Extremely high model fit. |
| **RMSE** | ~6.85 | Low deviation in USD terms. |
| **MAPE** | 0.15% | Very precise percentage error. |

## 🛠️ Project Structure
```bash
/Code
├── 01_fetch_news.py        # Fetches CryptoCompare news & calculates Sentiment (Loughran-McDonald)
├── 02_create_dataset.py    # Merges Price (Yahoo Finance) + Sentiment (Time-aligned UTC)
├── 03_EDA_Visualisasi.ipynb # Exploratory Data Analysis & Correlation Checks
├── 04_4H_data.ipynb        # Resampling to 4-Hour timeframe & Volatility Engineering
└── 05_Prediction_model.ipynb # Main LSTM Model Training & Evaluation
```

## 🔧 Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Arsyacoo/Artificial-Intellegence.git
    cd Artificial-Intellegence
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Pipeline**
    *   Step 1: Get News Data -> `python Code/01_fetch_news.py`
    *   Step 2: Build Dataset -> `python Code/02_create_dataset.py`
    *   Step 3: EDA & Visualization -> Open `Code/03_EDA_Visualisasi.ipynb`
    *   Step 4: Proccess 4H Data -> Open `Code/04_4H_data.ipynb`
    *   Step 5: Train Model -> Open `Code/05_Prediction_model.ipynb`



---
*Created by [Arsya] for Advanced AI Course (Semester 5).*
