# Bitcoin Trading Behavior & Market Sentiment Analysis
## Quick Start Guide

### 📋 Overview
This project analyzes the relationship between trader behavior on Hyperliquid and Bitcoin market sentiment (Fear & Greed Index) to predict profitable trading days using machine learning.

**Datasets:**
- Historical Trader Data: 211,000+ trading records
- Fear & Greed Index: 2,646 daily sentiment records (2018-present)

**ML Models Implemented:**
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Gradient Boosting

---

## 🚀 Getting Started

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the ML Models
Run the comprehensive analysis pipeline:
```bash
python ml_analysis_pipeline.py
```

**This will:**
- ✅ Load and clean both datasets
- ✅ Generate 10+ visualization charts (saved to `outputs/`)
- ✅ Engineer 17+ features for ML
- ✅ Train 5 different ML models with hyperparameter tuning
- ✅ Compare models and select the best performer
- ✅ Save the best model to `models/best_model.pkl`
- ✅ Create detailed performance reports

**Expected Runtime:** 10-30 minutes (depending on hardware)

### Step 3: Launch the Dashboard
Start the interactive Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📊 Dashboard Features

### 1. 🏠 Dashboard Page
- **Key Metrics:** Total trades, profitable trades, total P&L, average P&L
- **Charts:**
  - Market sentiment distribution (pie chart)
  - Profit distribution histogram
  - Sentiment value time series

### 2. 🔮 Make Predictions Page
- **Interactive Form:** Enter trading parameters
  - Trading volume metrics
  - Profit & price metrics
  - Market sentiment
  - Buy/sell ratio
- **Real-time Predictions:** Get instant profitability predictions
- **Probability Gauge:** Visual confidence indicator

### 3. 📊 Data Explorer Page
- **Trading Data Tab:** Filter and explore 211K+ trading records
- **Sentiment Data Tab:** View historical sentiment data
- **Download Feature:** Export filtered data as CSV

### 4. 📈 Model Insights Page
- **Performance Visualizations:**
  - Model comparison chart
  - Confusion matrix
  - ROC curves
  - Feature importance
- **Model Parameters:** Detailed configuration

---

## 🎯 Key Features

### Advanced Feature Engineering
- **Temporal Features:** Hour, day of week, month
- **Trading Metrics:** Volume, frequency, volatility
- **Profitability Metrics:** Average P&L, profit volatility
- **Behavioral Metrics:** Buy/sell ratio, trading patterns
- **Sentiment Integration:** Value and classification encoding

### Machine Learning Pipeline
- **Data Preprocessing:** Cleaning, merging, scaling
- **Feature Selection:** 17+ engineered features
- **Model Training:** 5 algorithms with GridSearchCV
- **Hyperparameter Tuning:** Optimized for maximum accuracy
- **Model Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC
- **Model Selection:** Automatic best model identification

### Visualizations Generated
1. Sentiment Distribution (Bar Chart)
2. Profit vs Sentiment (Box Plot)
3. Volume vs Sentiment (Violin Plot)
4. Correlation Matrix (Heatmap)
5. Profitability Rate by Sentiment (Bar Chart)
6. Sentiment Time Series (Line Chart)
7. Model Performance Comparison (Bar Chart)
8. Confusion Matrix (Heatmap)
9. ROC Curves (Line Chart)
10. Feature Importance (Bar Chart)

---

## 📁 Project Structure

```
ds_Ansh_Gupta/
├── csv_files/                      # Dataset files
│   ├── historical_data.csv         # 211K+ trading records
│   └── fear_greed_index.csv        # 2.6K sentiment records
├── outputs/                        # Visualization outputs
│   ├── 01_sentiment_distribution.png
│   ├── 02_profit_vs_sentiment.png
│   ├── ... (10 total charts)
│   └── 10_feature_importance.png
├── models/                         # Trained ML models
│   ├── best_model.pkl              # Best performing model
│   ├── scaler.pkl                  # Feature scaler
│   └── feature_columns.pkl         # Feature names
├── ml_analysis_pipeline.py         # Main ML pipeline
├── streamlit_app.py                # Interactive dashboard
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── USAGE_GUIDE.md                  # This file
```

---

## 🔬 Technical Details

### Target Variable
- **Prediction Task:** Binary classification (Profitable Day: Yes/No)
- **Definition:** Day is profitable if `Closed PnL > 0`

### Data Splits
- **Training Set:** 80% of data
- **Test Set:** 20% of data
- **Stratification:** Balanced target distribution

### Model Optimization
- **Cross-Validation:** 5-fold stratified K-fold
- **Hyperparameter Tuning:** GridSearchCV
- **Scoring Metric:** Accuracy (primary), also tracks Precision, Recall, F1, ROC-AUC

### Feature Scaling
- **Method:** StandardScaler (for Logistic Regression)
- **Application:** Fit on training data, transform on test data

---

## 💡 Usage Tips

### For Best Predictions
1. **Use Recent Data:** The model performs best with current market conditions
2. **Complete Information:** Fill all form fields for accurate predictions
3. **Realistic Values:** Use typical trading volumes and metrics
4. **Sentiment Alignment:** Match sentiment classification with sentiment value

### For Data Analysis
1. **Filter Wisely:** Use filters in Data Explorer to focus on specific patterns
2. **Time Periods:** Analyze different sentiment periods separately
3. **Volume Analysis:** High-volume days may show different patterns

### For Model Improvement
1. **More Data:** Add more recent trading records
2. **Feature Engineering:** Create new features from domain knowledge
3. **Model Tuning:** Adjust hyperparameters for your specific use case
4. **Ensemble Methods:** Combine multiple models for better performance

---

## 📈 Expected Performance

Based on the analysis pipeline:
- **Accuracy:** 60-80% (depends on data quality and market conditions)
- **Best Models:** Typically XGBoost or LightGBM
- **Feature Importance:** 
  - Sentiment value
  - Trading volume
  - Profit volatility
  - Buy/sell ratio

---

## 🐛 Troubleshooting

### Issue: Model files not found
**Solution:** Run `python ml_analysis_pipeline.py` first to train models

### Issue: Dashboard doesn't start
**Solution:** 
1. Check if Streamlit is installed: `pip install streamlit`
2. Verify you're in the correct directory
3. Try: `python -m streamlit run streamlit_app.py`

### Issue: Low model accuracy
**Solution:**
1. Ensure datasets are complete and correct
2. Check for data quality issues
3. Try different feature combinations
4. Adjust hyperparameter search ranges

### Issue: Visualizations not showing
**Solution:**
1. Ensure `outputs/` directory exists
2. Check if analysis pipeline completed successfully
3. Re-run the pipeline if needed

---

## 📞 Support

For questions or issues:
1. Review this usage guide
2. Check the README.md file
3. Verify all dependencies are installed
4. Ensure dataset files are in `csv_files/` directory

---

## 🎓 Learning Resources

### Understanding the Models
- **Logistic Regression:** Linear classification model
- **Random Forest:** Ensemble of decision trees
- **XGBoost:** Gradient boosted decision trees
- **LightGBM:** Fast gradient boosting framework
- **Gradient Boosting:** Sequential ensemble method

### Key Metrics Explained
- **Accuracy:** Overall correctness (correct predictions / total predictions)
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under the receiver operating characteristic curve

---

## 🔄 Next Steps

1. **Run the Pipeline:** Execute `ml_analysis_pipeline.py`
2. **Review Results:** Check `outputs/` folder for visualizations
3. **Launch Dashboard:** Start Streamlit app for predictions
4. **Analyze Patterns:** Explore how sentiment affects trading
5. **Refine Models:** Experiment with different features and algorithms

---

**📅 Last Updated:** February 10, 2026
**👤 Author:** Ansh Gupta
**🔖 Version:** 1.0
