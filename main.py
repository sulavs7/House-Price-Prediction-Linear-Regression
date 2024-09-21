import pandas as pd
import joblib
import numpy as np


# Loading trained model and scaler
housing_lr_model = joblib.load("housing_price_model.pkl")
scaler = joblib.load("scaler.pkl")

def preprocess_data(user_data):
    user_data=user_data.drop("price",axis=1)
    user_data=user_data.join(pd.get_dummies(user_data.furnishingstatus)).drop("furnishingstatus",axis=1)
    categorical_features = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnished','semi-furnished','unfurnished']
    for feature in categorical_features:
        user_data[feature] = user_data[feature].replace({'yes': 1, 'no': 0})
        user_data[feature] = user_data[feature].replace({True: 1,False: 0})
    user_data = user_data.infer_objects()  #replace bhanne ma warning aayera yo lekhya 
    return user_data  

def make_predictions(input_file):
    # Read user-uploaded dataset
    user_data = pd.read_csv(input_file)
    
    required_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']
    
    # done in eda of train data so doing here
    for feature in required_columns:
        user_data[feature] = np.log1p(user_data[feature])

    # to manage categorical features on data
    user_data = preprocess_data(user_data)

    # Scale the features
    user_data_scaled = scaler.transform(user_data)

    # Make predictions
    predictions = housing_lr_model.predict(user_data_scaled)

    # Inverse transform as in eda we have don log transform
    predictions = np.expm1(predictions)

    # Output results saved to seperate file 
    output_df = pd.DataFrame(predictions, columns=['Predicted Price'])
    output_df.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv succesfully!!")

if __name__ == "__main__":
    input_file = input("Enter the path to your dataset (CSV format): ")
    make_predictions(input_file)
