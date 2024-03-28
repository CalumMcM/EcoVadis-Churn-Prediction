from sklearn.model_selection import train_test_split

from data_analysis import DataAnalysis
from data_loader import DataLoader
from model import MachineLearningModel


def perform_data_analysis():
    """
    This function loads in the churn customer data and creates a DataAnalysis
    object for gaining insights to the data. 
    """
    dataloader = DataLoader("customer_data.xlsx")
    customer_data = dataloader.load_and_clean()

    data_analysis = DataAnalysis()

    # Split the data up into categorical and numerical data
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance (EUR)',
                      'NumberOfProducts', 'HasCreditCard', 'IsActiveMember',
                      'EstimatedSalary', 'Exited']

    categorical_cols = ['Surname', 'Country', 'Gender', 'CustomerFeedback']

    customer_cat = customer_data[categorical_cols]

    customer_num = customer_data[numerical_cols]

    # Output descriptions for both the numerical and categorical data
    print(f"Numerical Customer Data Description: {customer_num.describe()}\n")
    print(f"Categorical Customer Data Description: \
          {customer_cat.describe(exclude=['int64', 'float64'])}\n")

    # Get ratio of ground truth labels
    print(f"Ratio of exited vs stayed in data: \
          {customer_data['Exited'].value_counts()}\n")

    # Create a group box plot for numerical data, showing the mean and data
    # distribution
    cols_of_interest = ['CreditScore', 'Age', 'Tenure', 'Balance (EUR)',
                        'EstimatedSalary']

    data_analysis.group_box_plot(customer_data, cols_of_interest, num_cols=2)

    # Plot the distribution of estimated salaries for those that left vs stayed
    customer_data['EstimatedSalary'] = customer_data['EstimatedSalary']. \
        round(decimals=-4)
    title = "Retention Against Estimated Salary To Nearest Ten Thousand"
    save_name = "estimated_salary.png"
    data_analysis.compare_ratio_of_exited(customer_data, "EstimatedSalary",
                                               title, save_name)

    # Output mean age for those that have and have not left the company
    data_analysis.compare_mean_against_exited(customer_data, 'Age')


def perform_model_training():
    """
    This function loads in the churn customer data and creates a
    MachineLearningModel object in order to train a classifier on the data and
    evaluate its results. 
    """
    # Load and clean the data
    dataloader = DataLoader("customer_data.xlsx")
    customer_data = dataloader.load_and_clean()

    # Encode string type columns that can be easily done so
    cols_to_encode = ['Country', 'Gender']
    encoded_data = dataloader.apply_label_encoding(customer_data,
                                                   cols_to_encode)

    sentiment_data = dataloader.apply_sentiment_analysis(encoded_data,
                                                         ['CustomerFeedback'])

    # Remove columns that cannot be easily converted to a type the model can
    # extract meaningful information from
    drop_cols = ['RowNumber', 'CustomerId', 'Surname']
    reduced_data = sentiment_data.drop(columns=drop_cols)

    # Seperated labels from input data
    x = reduced_data.drop(columns=['Exited'])
    y = reduced_data['Exited']

    # Create train test splits
    x_train, x_test, y_train, y_test =  \
        train_test_split(x, y, test_size=0.3, random_state=42)

    # Apply SMOTE
    # oversample = SMOTE(k_neighbors=5)
    # x_smote, y_smote = oversample.fit_resample(x_train, y_train)
    # x_train, y_train = x_smote, y_smote
    # print (y_train.value_counts())

    machine_learning_model = MachineLearningModel(x_train, x_test, y_train,
                                                  y_test, "XGB")

    y_preds = machine_learning_model.fit_and_predict()

    machine_learning_model.evaluate(y_preds, y_test)


if __name__ == "__main__":

    perform_data_analysis()

    #perform_model_training()
