import math
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn import preprocessing


class DataLoader():
    """
    TODO: Class description, variables, methods
    TODO: Move class to own folder
    """
    
    def __init__(self, dataset_dir) -> None:
        self.dataset_dir = dataset_dir    
    
    def load_and_clean(self) -> pd.DataFrame:
        """
        Loads an excel spreadsheet from the global excel 
        directory

        Returns:
            * clean_dataframe: (pd.DataFrame) Pandas 
                dataframe of the given excel spreadsheet
                 that has had all NaN values errors replaced
        """

        excel_dataframe = pd.read_excel(self.dataset_dir)

        # Replaces all NaN values in an array with empty
        # strings. This is okay for this use case where the
        # only NaN values are in columns of type string.
        # However a more robust solution would check the
        # column type first the replace accordingly.
        clean_dataframe = excel_dataframe.fillna("")

        return clean_dataframe
    
    def apply_label_encoding(self, df, cols_to_apply) -> pd.DataFrame:
        """
        Takes a dataframe and a list of column headers 
        that should be encoding from their string values,
        and converts them to integers, e.g., 
        ["France", "Germany"] becomes [1,  2]
        
        Inputs:
            * df: (pd.DataFrame) The dataframe containing the
                data
            * cols_to_apply: (array(string)) The columns we 
                with to apply label encoding to

        Returns:
            * df: (pd.DataFrame) The original dataframe but
                the columns contained in the cols_to_apply
                input have been converted from strings to
                integer encodings
        """
        encoder = preprocessing.LabelEncoder()

        # Create a dataframe that just contains the columns of interest
        cols_of_interest = df[cols_to_apply]

        # Apply label encoding 
        cols_of_interest = cols_of_interest.apply(encoder.fit_transform)

        # Merge the updated columns of interest back to
        # the original dataset
        df[cols_to_apply] = cols_of_interest

        return df


class DataAnalysis():
    """
    TODO: Class description, variables, methods
    """

    def __init__(self) -> None:
        """
        Init function for this class
        """

    def compare_mean_against_existed(self, cust_data, col):
        """
        Takes a column name in the database and takes the mean
        of the value for both those that exited and those that
        remained. 

        TODO: Check column is float/int/double (not String)
        Inputs:     
            * cust_data: (pd.DataFrame) Dataframe that will be 
                analysed 
            * col: (string) Column that will be analysed

        Returns:
            * TODO:
        """

        # Get the data for both those who left and those
        # who stayed
        customer_stayed = cust_data.loc[cust_data['Exited'] == 0]
        customer_exited = cust_data.loc[cust_data['Exited'] == 1]

        all_mean = cust_data[col].mean()
        stayed_mean = customer_stayed[col].mean()
        left_mean = customer_exited[col].mean()

        # Output the results:
        print(f"Column: {col}\nAll: {all_mean}\n" \
              f"Stayed: {stayed_mean}\nLeft: {left_mean}\n")

        return all_mean, stayed_mean, left_mean
    
    def compare_mode_against_existed(self, cust_data, col, title, save_name):
        """
        Takes a column name in the database and counts the 
        number of occurances for each value. 

        Inputs:     
            * cust_data: (pd.DataFrame) Dataframe that will be 
                analysed 
            * col: (string) Column that will be analysed
            * title: (string) Title of the figure
            * save_name: (string) Name of figure to save

        Returns:
            * TODO
        """
        # Get the data for both those who left and those
        # who stayed
        customer_stayed = cust_data.loc[cust_data['Exited'] == 0]
        customer_exited = cust_data.loc[cust_data['Exited'] == 1]

        # Calculate the mode
        all_counts = cust_data[col].value_counts()
        stayed_counts = customer_stayed[col].value_counts()
        left_counts = customer_exited[col].value_counts()

        df_combined = pd.DataFrame()

        df_combined['All'] = all_counts
        df_combined['Stayed'] = stayed_counts
        df_combined['Left'] = left_counts

        # Calculate ratio of those that left vs stayed
        country_ratios = []

        # TODO: Use something other than iterrows for efficiency
        for _, row in df_combined.iterrows():
            ratio_left = row['Left']/row['All']
            ratio_stayed = row['Stayed']/row['All']
            country_ratios.append([100, ratio_stayed, ratio_left, row['All']])

        ratios_dataframe = pd.DataFrame(country_ratios)
        print (f"Ratios: {ratios_dataframe}")

        df_combined.sort_index(inplace=True)
        ax = df_combined.plot(kind='bar')

        ax.set_title(title)
        ax.set_ylabel("Num. Customers")
        # Output the result
        print (df_combined)

        plt.tight_layout()
        plt.savefig("Figures/"+save_name)
        plt.show()

        return df_combined

    def group_box_plot(self, df, cols_of_interest, num_cols):
        """
        Takes a list of column names of interest and a dataframe
        that they are within. A group of boxplots are then made
        for each of these columns of interest against whether 
        or not the user has exited from the program. 

        Inputs:
            * df: (pd.DataFrame) Dataframe containing the 
                cols_of_interest
            * cols_of_interest: (array(String)) An array 
                containing the column names of interest
                for the given dataframe
            * num_cols: (int) The number of columns that
                will be in the final plot
        """
        # For number of rows in subplot, divide the number
        # of columns we are interested in by number of columns
        # and then round up
        num_rows = math.ceil(len(cols_of_interest) / num_cols)
        
        _, axes = plt.subplots(num_rows, num_cols, sharex=True, 
                               figsize=(7,10))
        
        row, col = 0, 0
        for column_name in cols_of_interest:
            
            sns.boxplot(ax=axes[row,col], x='Exited', 
                        y=column_name, data=df)
            
            axes[row, col].set_title(f"{column_name} vs Exited")

            # Move to next column, if last column then
            # wrap to next row
            col += 1
            if col > num_cols-1:
                col = 0
                row += 1
                        
        plt.tight_layout()
        plt.savefig("Figures/group_box_plot.png")
        plt.show()   


def main():

    dataloader = DataLoader("customer_data.xlsx")
    customer_data = dataloader.load_and_clean()

    data_analysis = DataAnalysis()
    
    cois = ['Country', 'Gender']
    customer_data = dataloader.apply_label_encoding(customer_data, cois)
    

    #numerical_cols = ['CreditScore' , 'Age', 'Tenure', 'Balance (EUR)', 'NumberOfProducts', #'HasCreditCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']

    #customer_cat = customer_data[['Surname', 'Country', 'Gender', 'CustomerFeedback']]
    #customer_num = customer_data[numerical_cols]

    #print (customer_num.describe())

    #print (customer_cat.describe(exclude=['int64', 'float64']))
    
    # Get ratio of ground truth labels
    #print (customer_data['Exited'].value_counts())

    # Create a group box plot for numerical data
    cols_of_interest = ['CreditScore' , 'Age', 'Tenure', 'Balance (EUR)', 'EstimatedSalary']
    data_analysis.group_box_plot(customer_data, cols_of_interest, num_cols=2)

    """
    customer_data['EstimatedSalary'] = customer_data['EstimatedSalary'].round(decimals=-4)
    title = "Retention Against Estimated Salary To Nearest Ten Thousand"
    save_name = "estimated_salary.png"
    data_analysis.compare_mode_against_existed(customer_data, "EstimatedSalary", title, save_name)
    """

    #data_analysis.compare_mean_against_existed(customer_data, 'Age')


if __name__ == "__main__":
    main()