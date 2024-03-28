"""
This module contains the MachineLearningModel class that allows a user to make
predictions with a desired classification model and evaluate the results. 
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


class MachineLearningModel():
    """
    This class allows a user to train either a Random Forest Classifier or
    a XGBoost classifier with a given set of training and test data. The user
    can then evaluate this model by creating a confusion matrix and
    classification report against the predictions made by the model. 

    Attributes:
        * self.x_train: (pd.DataFrame) 
        * self.x_test: (pd.DataFrame)  
        * self.y_train: (pd.DataFrame)  
        * self.y_test: (pd.DataFrame)  
        * self.model_type: (string) 
        * model_type: (string) 
        * self.model: (RandomForestClassifier or xgb.XGBRegressor)

    Methods:
        * fit_and_predict
        * evaluate
    """

    def __init__(self, x_train, x_test,
                 y_train, y_test, model_type) -> None:
        """
        Init function for the class

        Inputs:
            * x_train: (pd.DataFrame) Input training
                data for the model
            * x_test: (pd.DataFrame) Input test data
                for the model
            * y_train: (pd.DataFrame) Input training
                labels for the model
            * y_test: (pd.DataFrame) Input test labels
                for the model
            *model_type: (string) Allows for different
                model types to be evaluated. Allowed
                options are:
                    - "RF" for RandomForestClassifier
                    - "XGB" for XGBoost
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_type = model_type
        # Set model to chosen type
        if model_type == "RF":
            self.model = RandomForestClassifier(random_state=42)
        elif model_type == "XGB":
            self.model = xgb.XGBRegressor(objective="binary:logistic")
        else:
            self.model = None
            print("No model chosen! The selected model \
                    must be either RF or XGB.")

    def fit_and_predict(self):
        """
        Trains the chosen model type on the training data
        and returns predictions for the test set. 

        Returns:
            * y_pred: (array(int))
        """
        self.model.fit(self.x_train, self.y_train)

        y_pred = self.model.predict(self.x_test)

        # If XGB classifier is used then convert
        # predictions around 0.5 decision boundary
        if self.model_type == "XGB":
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0

        return y_pred

    def evaluate(self, y_preds, y_test):
        """
        Plots of a confusion matrix with a given 
        set of predictions and outputs the 
        classifcation report to the terminal.

        Inputs:
            * y_preds: (array(int)) Predictions
                by the model of whether or not
                a user has stayed (0) or exited (1)
            * y_test: (array(int)) Ground truth
                labels for each of the predictions
        """
        matrix = confusion_matrix(self.y_test, y_preds)
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

        # Build the plot
        plt.figure(figsize=(16, 7))
        sns.heatmap(matrix, annot=True, annot_kws={'size': 10},
                    cmap=plt.cm.Greens, linewidths=0.2)

        # Add labels to the plot
        class_names = ['Stayed', 'Exited']
        tick_marks = np.arange(len(class_names))+0.5
        plt.xticks(tick_marks, class_names, rotation=0)
        plt.yticks(tick_marks, class_names, rotation=0)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title(f'Confusion Matrix for {self.model_type}')
        plt.savefig(f'./Figures/conf_matrix_SENTIMENT_{self.model_type}')
        plt.show()

        # Output the classification report
        print(classification_report(y_test, y_preds))
