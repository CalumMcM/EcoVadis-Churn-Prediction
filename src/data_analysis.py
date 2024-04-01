"""
This module contains the DataAnalysis class which allows a user to create a
variety of plots based on inputted data. 
"""
import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class DataAnalysis():
    """
    This class allows a user to create graphical evaluations of input data. 

    Methods:
        * compare_mean_against_exited
        * compare_ratio_of_exited
        * group_box_plot
    """

    def compare_mean_against_exited(self, cust_data: pd.DataFrame,
                                    col: str) -> tuple[float, float, float]:
        """
        Takes a column name in the database and takes the mean of the value for
        both those that exited and those that remained. This allows for
        individual comparison of those that left and stayed for indivdual
        categories.

        Inputs:     
            * cust_data: (pd.DataFrame) Dataframe that will be 
                analysed 
            * col: (string) Column that will be analysed

        Returns:
            * all_mean: (float) The mean value for everyone that stayed and left
            * stayed_mean: (float) The mean value for those that stayed
            * left_mean: (flaot) The mean value for those that exited
        """

        # Get the data for both those who left and those
        # who stayed
        customer_stayed = cust_data.loc[cust_data['Exited'] == 0]
        customer_exited = cust_data.loc[cust_data['Exited'] == 1]

        # Calculate the means for each of the three output categories
        all_mean = cust_data[col].mean()
        stayed_mean = customer_stayed[col].mean()
        left_mean = customer_exited[col].mean()

        # Output the results:
        print(f"\nColumn: {col}\nAll: {all_mean}\n"
              f"Stayed: {stayed_mean}\nLeft: {left_mean}\n")

        return all_mean, stayed_mean, left_mean

    def compare_splits_against_col(self, splits: dict[str: pd.DataFrame],
                                   col: str, title: str, save_name: str) \
            -> pd.DataFrame:
        """ 
        Given two splits of a dataset, computes the ratio of those two splits 
        against a given column that they both share.  

        Inputs:     
            * splits: (dict) Dictionary containing two splits of a dataset in
              the format of {'split_name' : pd.DataFrame}
            * col: (string) Column that will be analysed
            * title: (string) Title of the figure
            * save_name: (string) Name of figure to save

        Returns:
            * df_combined: (pd.DataFrame) The number of occurrences for each of
              the three categories of interest. 
        """
        # Extract the names and splits from the given dictionary
        split_names = list(splits.keys())
        split1 = splits[split_names[0]]
        split2 = splits[split_names[1]]

        # Calculate the number of occurrences for those that left, those that
        # stayed and both
        df_combined = pd.DataFrame()
        df_combined[split_names[0]] = split1[col].value_counts()
        df_combined[split_names[1]] = split2[col].value_counts()
        df_combined['All'] = split1[col].value_counts() + \
            split2[col].value_counts()

        # Calculate ratio of those that left vs stayed
        all_row_ratios = []
        for _, row in df_combined.iterrows():
            ratio1 = row[split_names[0]]/row['All']
            ratio2 = row[split_names[1]]/row['All']
            all_row_ratios.append([ratio1.round(2), ratio2.round(2)])

        ratios_dataframe = pd.DataFrame(all_row_ratios)
        print(f"Ratios: \n{ratios_dataframe}\n")

        # Remove all data from bar chart
        df_combined = df_combined.drop(columns="All")

        # Output the results as a bar chart
        df_combined.sort_index(inplace=True)
        ax = df_combined.plot(kind='bar', color=['#00C43C', '#E95238'])

        ax.set_title(title)
        ax.set_ylabel("Num. Customers")
        # print the result
        print(df_combined)

        plt.tight_layout()
        # Commenting out savefig for Docker Container
        # plt.savefig("Figures/"+save_name)
        plt.show()

        return df_combined

    def group_box_plot(self, df: pd.DataFrame, cols_of_interest: list[str],
                       num_cols: int):
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
                               figsize=(7, 10))

        row, col = 0, 0
        for column_name in cols_of_interest:

            sns.boxplot(ax=axes[row, col], x='Exited', y=column_name, data=df,
                        palette="Greens", hue='Exited', legend=False)

            axes[row, col].set_title(f"{column_name} vs Exited")

            # Move to next column, if last column then
            # wrap to next row
            col += 1
            if col > num_cols-1:
                col = 0
                row += 1

        plt.tight_layout()
        # plt.savefig("Figures/group_box_plot.png")
        plt.show()
