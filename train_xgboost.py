import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load features from MT5 Files folder
features_file = "C:/Users/Administrator/AppData/Roaming/MetaQuotes/Terminal/D0E8209F77C8CF37AD8BF550E51FF075/MQL5/Files/FXJEFE_Features.csv"
if not pd.io.common.file_exists(features_file):
    print(f"Error: {features_file} not found. Run the EA to generate it.")
    exit()

df = pd.read_csv(features_file)
print(f"Loaded {len(df)} rows from {features_file}")

# Features used for prediction (match EA inputs to CallAIAPI)
X = df[['price', 'atr', 'ema_diff', 'rsi', 'garch_vol', 'macd_diff', 'vwap', 'vwap_upper', 'vwap_lower']]
y = df['signal']  # Target: 1 (buy), -1 (sell), 0 (hold)

# Check data
if len(X) < 10 or y.nunique() < 2:
    print("Insufficient data or signal variety for training. Need more data with buy/sell signals.")
    exit()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,  # 1, -1, 0
    eval_metric='mlogloss',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100
)
model.fit(X_train, y_train)
print("Model training completed.")

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['sell', 'hold', 'buy']))

# Save the model
model.save_model("fxjefe_xgboost_model.json")
print("Model saved to fxjefe_xgboost_model.json")

