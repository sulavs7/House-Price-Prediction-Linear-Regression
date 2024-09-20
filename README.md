# House Price Prediction - Linear Regression

This project is a simple house price prediction model using Linear Regression. It allows users to input their dataset and get predictions for house prices based on several features.

## Features
- Predicts house prices based on multiple features (e.g., area, bedrooms, bathrooms, stories, parking, etc.)
- Includes pre-trained model and scaling files
- Users can input their own datasets in CSV format

## Files in the Repository

- `models/main.py`: Main module where users can input their dataset and check the model predictions.
- `models/evaluation.py`: A script that calculates evaluation metrics for the model.
- `models/housing_price_model.pkl`: Pre-trained model file.
- `models/scaler.pkl`: Scaler used to transform the dataset before prediction.
- `Data/housing-test-set.csv`: Example dataset for testing.
- `predictions.csv`: CSV file where predicted values are saved.
- `requirements.txt`: List of required dependencies for the project.

## Requirements
To run this project, you need to have the following libraries installed:

```txt
pandas==1.5.3
scikit-learn==1.2.0
joblib==1.2.0
numpy==1.24.2
```

## You can install these using the requirements.txt file by following command mentioned bellow in your terminal:
```txt
pip install -r requirements.txt
```
## How to Clone the Repository
To clone this repository, run the following command in your terminal:
```txt
git clone https://github.com/sulavs7/House-Price-Prediction-Linear-Regression.git
```

Once cloned, navigate to the project directory:
```txt
cd House-Price-Prediction-Linear-Regression
```

## How to Use
1.Make sure the required packages are installed (pandas, scikit-learn, etc.).
2.Run the main.py script in the models folder.
3.Enter the path to your dataset when prompted. The predictions will be saved to predictions.csv.


## License
This project is licensed under the MIT License.


