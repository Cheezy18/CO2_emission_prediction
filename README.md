# CO₂ Emissions Prediction using Machine Learning

## 📊 Project Overview

This project implements and compares three gradient boosting and ensemble machine learning algorithms to predict annual CO₂ emissions based on historical data. The goal is to understand emission patterns across different countries and predict future emissions using advanced regression techniques.

## 📁 Dataset Information

**Dataset**: CO₂ Emissions Dataset  
**Source**: `co2_emission.csv`

### Original Features
- **Entity**: Country name
- **Code**: Country code (ISO 3166-1 alpha-3)
- **Year**: Year of observation
- **Annual CO₂ emissions (tonnes)**: Target variable

### Dataset Statistics
- **Total Records**: 18,379 (after preprocessing)
- **Training Set**: 14,703 samples (80%)
- **Test Set**: 3,676 samples (20%)
- **Number of Countries**: 222
- **Final Features**: 227 (after feature engineering and encoding)

## 🔧 Data Preprocessing Pipeline

### 1. Data Cleaning
- Dropped rows with missing `Code` values
- Filtered out non-country entities (e.g., regions, continents)
- Outlier capping at 99th percentile to handle extreme values

### 2. Feature Engineering
- **Historical_Avg_CO2**: Rolling average of CO₂ emissions per country
  - Captures long-term emission trends for each entity
  - Helps model understand historical patterns
  
- **CO2_Change**: Year-over-year change in CO₂ emissions
  - Measures emission growth rate
  - Identifies acceleration or deceleration in emissions
  
- **Year_Avg_Interaction**: Interaction between Year and Historical_Avg_CO2
  - Captures non-linear relationships
  - Models time-dependent emission patterns

### 3. Transformations
- **Log Transformation**: Applied to target variable to handle skewed distribution
- **One-Hot Encoding**: Converted categorical `Entity` feature into 222 binary features
- **StandardScaler**: Normalized `Year` and target variable for better model convergence

## 🤖 Models Implemented

### 1. Random Forest Regressor (Hyperparameter Tuned)

#### Configuration
```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
```

#### Hyperparameter Tuning
- **Method**: GridSearchCV with 5-fold cross-validation
- **Search Space**: Tested various combinations of n_estimators, max_depth, min_samples_split, and min_samples_leaf
- **Scoring Metric**: Negative Mean Squared Error
- **Best Parameters**: n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1

#### Performance Metrics

**Log-Transformed Scale**:
- MAE: 0.0595
- MSE: 0.0143
- RMSE: 0.1196
- **R² Score: 0.9861**

**Original Scale**:
- MAE: 0.06 tonnes
- MSE: 0.01
- RMSE: 0.11 tonnes
- **R² Score: 0.9926**

#### ✅ Pros
- **High Interpretability**: Feature importance scores readily available
- **Robust to Outliers**: Ensemble of trees reduces sensitivity to anomalies
- **No Feature Scaling Required**: Works well with raw features (though we scaled for consistency)
- **Handles Non-linearity**: Captures complex relationships without explicit feature engineering
- **Excellent Generalization**: Best R² on original scale (0.9926) indicates strong real-world performance
- **Reduced Overfitting**: Tuned hyperparameters prevent overfitting
- **Parallel Processing**: Fast training with n_jobs=-1

#### ❌ Cons
- **Higher Computational Cost**: 200 trees require more memory and training time
- **Slower Predictions**: Ensemble predictions slower than single models
- **Moderate Log-Scale Performance**: Lowest R² on log scale (0.9861) among all models
- **Large Model Size**: 200 trees create large model file, challenging for deployment
- **Black Box Nature**: Individual tree decisions difficult to interpret despite feature importance
- **Parameter Sensitivity**: Requires careful tuning for optimal performance

---

### 2. XGBoost Regressor

#### Configuration
```python
XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    n_jobs=-1
)
```

#### Performance Metrics

**Log-Transformed Scale**:
- MAE: 0.0285
- MSE: 0.0052
- RMSE: 0.0722
- **R² Score: 0.9954** ⭐ **Best Overall**

**Original Scale**:
- MAE: 9.77 tonnes
- MSE: 35920.03
- RMSE: 142.29 tonnes
- **R² Score: 0.9954**

#### ✅ Pros
- **Best Overall Performance**: Highest R² score (0.9954) on both scales
- **Lowest Log-Scale Errors**: Best MAE (0.0285), MSE (0.0052), and RMSE (0.0722)
- **Regularization Built-in**: L1 and L2 regularization prevent overfitting
- **Handles Missing Data**: Native support for missing values
- **Feature Importance**: Provides multiple importance metrics (gain, cover, frequency)
- **Efficient Training**: Gradient boosting with optimized algorithms
- **Balanced Complexity**: Max_depth=3 provides good bias-variance tradeoff
- **Industry Standard**: Widely used in competitions and production

#### ❌ Cons
- **Higher Original Scale MAE**: MAE of 9.77 tonnes vs Random Forest's 0.06
- **Sensitive to Hyperparameters**: Performance heavily dependent on tuning
- **Requires Careful Tuning**: Learning rate, max_depth, and n_estimators need optimization
- **Sequential Training**: Cannot fully parallelize like Random Forest
- **Overfitting Risk**: Can overfit with too many estimators or deep trees
- **Memory Intensive**: Stores gradients and hessians during training
- **Less Robust to Outliers**: Compared to Random Forest

---

### 3. LightGBM Regressor

#### Configuration
```python
LGBMRegressor(
    objective='regression',
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
```

#### Performance Metrics

**Log-Transformed Scale**:
- MAE: 0.0284
- MSE: 0.0064
- RMSE: 0.0801
- **R² Score: 0.9944**

**Original Scale**:
- MAE: 28.27 tonnes
- MSE: 192083.94
- RMSE: 438.27 tonnes
- **R² Score: 0.9565** ⚠️ **Lowest**

#### ✅ Pros
- **Fastest Training**: Histogram-based algorithm significantly faster than XGBoost
- **Low Log-Scale MAE**: Comparable to XGBoost (0.0284 vs 0.0285)
- **Memory Efficient**: Uses histogram bins instead of pre-sorted features
- **Handles Large Datasets**: Optimized for datasets with millions of rows
- **Leaf-wise Growth**: Can achieve better accuracy with fewer trees
- **Native Categorical Support**: Can handle categorical features directly
- **Good Log-Scale Performance**: R² of 0.9944 still excellent

#### ❌ Cons
- **Poor Original Scale Performance**: Lowest R² (0.9565) on original scale
- **Highest Original Scale Errors**: MAE of 28.27 and RMSE of 438.27 tonnes
- **Large Prediction Variance**: Significant discrepancy between log and original scale
- **Overfitting Tendency**: Leaf-wise growth can overfit small datasets
- **Requires More Tuning**: Needs careful parameter tuning to match XGBoost
- **Less Stable Predictions**: Higher variance in inverse-transformed predictions
- **Not Optimal for This Dataset**: Better suited for larger datasets

---

## 📈 Model Comparison Summary

| Model | Log R² | Original R² | Log MAE | Original MAE | Training Speed | Best For |
|-------|--------|-------------|---------|--------------|----------------|----------|
| **Random Forest (Tuned)** | 0.9861 | **0.9926** ⭐ | 0.0595 | **0.06** ⭐ | Moderate | Real-world predictions |
| **XGBoost** | **0.9954** ⭐ | 0.9954 | **0.0285** ⭐ | 9.77 | Moderate | Overall accuracy |
| **LightGBM** | 0.9944 | 0.9565 | 0.0284 | 28.27 | **Fastest** ⭐ | Large datasets |

### Key Insights

1. **Best Overall Model**: **XGBoost** 
   - Highest R² score (0.9954) on log scale
   - Best balance of accuracy and complexity
   - Recommended for production deployment

2. **Best Real-World Predictions**: **Random Forest (Tuned)**
   - Lowest MAE on original scale (0.06 tonnes)
   - Most reliable for actual emission predictions
   - Best for interpretability needs

3. **Fastest Training**: **LightGBM**
   - Ideal for rapid prototyping
   - Not recommended due to poor original scale performance
   - Needs further hyperparameter tuning

## 🔍 Technical Considerations

### Why Log Transformation?
- CO₂ emissions span several orders of magnitude
- Log transformation normalizes distribution
- Improves model convergence and stability
- Reduces impact of extreme outliers

### Inverse Transformation Challenges
- **Random Forest**: Excellent inverse transformation (R²: 0.9926)
- **XGBoost**: Good inverse transformation (R²: 0.9954)
- **LightGBM**: Poor inverse transformation (R²: 0.9565)
  - Indicates prediction variance issues
  - Suggests overfitting on log scale

### Feature Importance
All models benefit from:
- **Entity features**: Country-specific patterns critical
- **Historical_Avg_CO2**: Strong predictor of future emissions
- **Year**: Temporal trends important
- **CO2_Change**: Captures momentum in emissions

## 🚀 Usage

### Prerequisites
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

### Training Pipeline
1. Load and clean data
2. Engineer features (Historical_Avg_CO2, CO2_Change, Year_Avg_Interaction)
3. Apply log transformation to target
4. One-hot encode countries
5. Scale numerical features
6. Train models
7. Evaluate on both log and original scales

## 📊 Visualizations Generated
- Actual vs. Predicted scatter plots for all models
- Residual plots
- Feature importance charts
- Model comparison bar charts

## 🎯 Recommendations

### For Production Deployment
**Use XGBoost** for the best balance of accuracy and reliability.

### For Interpretability
**Use Random Forest (Tuned)** with feature importance analysis.

### For Speed/Prototyping
**Use LightGBM** but with extensive hyperparameter tuning.

### For Future Improvements
1. **Hyperparameter Tuning**: Apply GridSearchCV to XGBoost and LightGBM
2. **Feature Selection**: Remove low-importance country features
3. **Ensemble Methods**: Combine predictions from all three models
4. **Deep Learning**: Explore LSTM for time-series patterns
5. **Additional Features**: Include GDP, population, industrial indicators



## 👤 Author

Selvaraghavan S
[Github:@Cheezy18](https://github.com/Cheezy18/CO2_emission_prediction)

---

**Note**: All metrics reported are based on a fixed 80/20 train-test split with random_state=42 for reproducibility.
