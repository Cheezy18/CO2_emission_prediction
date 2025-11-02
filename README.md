CO2 Emission Prediction Project
Goal
To prepare the CO2 emission dataset for machine learning by explicitly addressing three critical challenges inherent to this data type: time dependency, data skewness, and entity imbalance.

Key Methodology Highlights
The data pipeline was designed to transform the raw data into a format optimal for high-performance regression models (e.g., Random Forest or XGBoost).

Time Series Aspect: Handled by creating Lagged Features to give the model historical memory.

Skewed Distribution (CO2 Target): Corrected by applying a Log(1+x) Transformation to normalize the target variable.

Entity Imbalance: Addressed by using One-Hot Encoding on the 'Country' feature.

Data Preprocessing & Feature Engineering Pipeline
1. Data Cleaning
Entity Filtering: Aggregated non-country entities (e.g., 'World', 'Africa', 'G20') were removed.

Target Cleaning: Rows with missing or non-positive CO2 emission values (specifically 40 negative rows) were removed to enable the required logarithmic transformation.

2. Time-Series Feature Engineering
Two key time-series features were created for each country:

Historical_Avg_CO2 (Trend): The expanding mean of all previous years' CO2 emissions.

CO2_Change (Momentum): The year-over-year difference in CO2 emissions.

3. Target Variable Transformation (Critical Step)
The raw target variable, Y, was transformed using the log(1+x) function to create the new, normally distributed target: Y' = log(1+Y).

4. Final Data Preparation
Feature Scaling: Numeric features (Year, Historical_Avg_CO2, CO2_Change) were scaled using StandardScaler.

Categorical Encoding: The Country column was converted into binary features using OneHotEncoder for entity-specific modeling.

Train/Test Split: The dataset was split into 80% Training and 20% Testing sets, ready for the modeling stage.
