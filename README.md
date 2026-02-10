# Bitcoin Trading Behavior & Market Sentiment Analysis

## Project Overview
This project analyzes the relationship between trader behavior on Hyperliquid and Bitcoin market sentiment (Fear & Greed Index). The goal is to identify patterns and build predictive models that can inform smarter trading strategies.

## Directory Structure
```
ds_Ansh_Gupta/
├── notebook_1.ipynb          # Main analysis and ML model development
├── notebook_2.ipynb          # Additional analysis (if needed)
├── csv_files/                # Dataset files
│   ├── trader_data.csv       # Historical trader data from Hyperliquid
│   └── sentiment_data.csv    # Bitcoin Fear & Greed Index
├── outputs/                  # Visualizations and charts
│   └── *.png / *.jpg
├── models/                   # Trained ML models
│   └── best_model.pkl
├── streamlit_app.py          # Interactive dashboard
├── ds_report.pdf             # Final insights report
└── README.md                 # This file
```

## Datasets

### 1. Bitcoin Market Sentiment Dataset
- **Columns**: Date, Classification (Fear / Greed)
- **Source**: Fear & Greed Index

### 2. Historical Trader Data from Hyperliquid
- **Columns**: account, symbol, execution price, size, side, time, start position, event, closedPnL, leverage, etc.
- **Source**: Hyperliquid trading platform

## Setup Instructions

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step 1: Train the ML Models
Run the comprehensive analysis pipeline:
```bash
python ml_analysis_pipeline.py
```

This will:
- Load and process 211K+ trading records and 2.6K sentiment records
- Generate 10+ visualization charts in the `outputs/` folder
- Train 5 ML models (Logistic Regression, Random Forest, XGBoost, LightGBM, Gradient Boosting)
- Perform hyperparameter tuning with GridSearchCV
- Save the best model to `models/best_model.pkl`

**Expected runtime:** 10-30 minutes

### Step 2: Launch the Interactive Dashboard
```bash
streamlit run streamlit_app.py
```

The dashboard will open at `http://localhost:8501` with 4 interactive pages:
- 🏠 **Dashboard:** Key metrics and visualizations
- 🔮 **Predictions:** Make real-time profitability predictions
- 📊 **Data Explorer:** Filter and explore datasets
- 📈 **Model Insights:** View model performance and feature importance

### Running the Analysis
1. Place the dataset files in the `csv_files/` directory
2. Open `notebook_1.ipynb` in Jupyter Notebook or Google Colab
3. Run all cells to perform EDA, feature engineering, and model training

### Running the Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```

## Key Objectives
1. **Analyze Trading Behavior**: Understand profitability, risk, volume, and leverage patterns
2. **Sentiment Alignment**: Explore how trader behavior aligns with market sentiment
3. **Predictive Modeling**: Build high-accuracy ML models to predict trading outcomes
4. **Strategic Insights**: Identify hidden trends and signals for smarter trading

## Model Performance
- See `notebook_1.ipynb` for detailed model evaluation metrics
- Best performing model saved in `models/best_model.pkl`

## Key Findings
- Detailed insights available in `ds_report.pdf`
- Visualizations available in `outputs/` directory

## Author
Ansh Gupta

## Last Updated
February 10, 2026
