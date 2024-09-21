import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_predictions(y_true, y_pred):
    """
    Evaluate the performance of a regression model.

    Parameters:
    y_true (array-like): True values of the target variable.
    y_pred (array-like): Predicted values of the target variable.

    Returns:
    dict: Dictionary containing MAE, MSE, RMSE, and R².
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }
    
    return metrics


def evaluate_classification(y_true, y_pred):
    """
    Evaluate the performance of a classification model.

    Parameters:
    y_true (array-like): True values of the target variable.
    y_pred (array-like): Predicted values of the target variable.

    Returns:
    dict: Dictionary containing Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': cm
    }
    
    return metrics

# Example usage
if __name__ == "__main__":
    # Example true and predicted values for housing prices
    # For using evaluate_predictions() in other file use import as:
    # from evaluate_classification import evaluate_classification
    y_true = np.array([300000, 150000, 250000, 450000, 200000])
    y_pred = np.array([310000, 140000, 260000, 440000, 210000])
    
    metrics = evaluate_predictions(y_true, y_pred)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        
    
    # Example true and predicted values for wine quality
    # For using evaluate_classification() in other file use import as:
    # from evaluate_classification import evaluate_classification
    y_true = [3, 2, 2, 2, 3, 3, 1, 1, 2, 3]
    y_pred = [3, 2, 2, 2, 1, 3, 1, 2, 2, 3]
    
    metrics = evaluate_classification(y_true, y_pred)
    for metric, value in metrics.items():
        if metric != 'Confusion Matrix':
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}:\n{value}")
