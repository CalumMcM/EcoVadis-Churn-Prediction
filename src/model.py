from sklearn.model_selection import train_test_split
from data_analysis import DataLoader
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, \
    classification_report
from imblearn.over_sampling import SMOTE

import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


class MachineLearningModel():
    """
    TODO: Write summary
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
            print ("No model chosen! The selected model \
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

    def conf_matrix(self, y_preds):
        """
        Plots of a confusion matrix with a given 
        set of predictions. 

        Inputs:
            * y_preds: (array(int)) Predictions
                by the model of whether or not
                a user has stayed (0) or exited (1)
        """
        matrix = confusion_matrix(self.y_test, y_preds)
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

        # Build the plot
        plt.figure(figsize=(16,7))
        sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                    cmap=plt.cm.Greens, linewidths=0.2)

        # Add labels to the plot
        class_names = ['Stayed', 'Exited']
        tick_marks = np.arange(len(class_names))+0.5
        plt.xticks(tick_marks, class_names, rotation=0)
        plt.yticks(tick_marks, class_names, rotation=0)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title(f'Confusion Matrix for {self.model_type}')
        plt.savefig(f'./Figures/conf_matrix_SMOTE_{self.model_type}')
        plt.show()

def main():

    # Load and clean the data
    dataloader = DataLoader("customer_data.xlsx")
    customer_data = dataloader.load_and_clean()

    # Encode string type columns that can be easily done so
    cols_to_encode = ['Country', 'Gender']
    encoded_data = dataloader.apply_label_encoding(customer_data, cols_to_encode)

    # Remove columns that cannot be easily converted
    # to a type the model can extract
    # meaningful information from
    drop_cols = ['RowNumber', 'CustomerId', 'Surname', 'CustomerFeedback']
    reduced_data = encoded_data.drop(columns=drop_cols)

    # Seperated labels from input data
    x = reduced_data.drop(columns=['Exited'])
    y = reduced_data['Exited']
    
    # Create train test splits
    x_train, x_test, y_train, y_test =  \
                train_test_split(x, y, test_size=0.3, random_state=42)
    
    # Apply SMOTE 
    #oversample = SMOTE(k_neighbors=5)
    #x_smote, y_smote = oversample.fit_resample(x_train, y_train)
    #x_train, y_train = x_smote, y_smote
    #print (y_train.value_counts())
    
    machine_learning_model = MachineLearningModel(x_train, x_test, y_train,
                                                  y_test, "XGB")
    
    y_preds = machine_learning_model.fit_and_predict()

    machine_learning_model.conf_matrix(y_preds)

    print(classification_report(y_test, y_preds))


if __name__ == "__main__":

    main()