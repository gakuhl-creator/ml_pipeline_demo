import joblib

model = joblib.load('../models/random_forest_churn.pkl')
print("Trained feature names:")
print(model.feature_names_in_)
