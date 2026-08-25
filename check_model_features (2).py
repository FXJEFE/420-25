import joblib
model = joblib.load('C:\\Users\\Administrator\\Documents\\FXJEFE_Project\\models\\my_model.pkl')
print(f"Model expects {model.n_features_in_} features")