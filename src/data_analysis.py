"""
This module contains the DataAnalysis class which allows a user to create a
variety of plots based on inputted data. 
"""
import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


class DataAnalysis():
    """
    This class allows a user to create graphical evaluations of input data. 

    Methods:
        * compare_mean_against_existed
        * compare_ratio_of_exited
        * group_box_plot
    """

    def compare_mean_against_existed(self, cust_data, col):
        """
        Takes a column name in the database and takes the mean of the value for
        both those that exited and those that remained. This allows for
        individual comparison of those that left and stayed for indivdual
        categories.

        TODO: Check column is float/int/double (not String) Inputs:     
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

    def compare_ratio_of_exited(self, cust_data, col, title, save_name):
        """
        Takes a column name in the database and counts the number of occurances
        and ratios for the three categories of: 'All', 'Stayed', and 'Exited'. 

        Inputs:     
            * cust_data: (pd.DataFrame) Dataframe that will be analysed
            * col: (string) Column that will be analysed
            * title: (string) Title of the figure
            * save_name: (string) Name of figure to save

        Returns:
            * df_combined: (pd.DataFrame) The number of occurances for each of
              the three categories of interest. 
        """
        # Get the data for both those who left and those
        # who stayed
        customer_stayed = cust_data.loc[cust_data['Exited'] == 0]
        customer_exited = cust_data.loc[cust_data['Exited'] == 1]

        # Calculate the number of occurances for three categories
        df_combined = pd.DataFrame()

        df_combined['All'] = cust_data[col].value_counts()
        df_combined['Stayed'] =  customer_stayed[col].value_counts()
        df_combined['Left'] = customer_exited[col].value_counts()

        # Calculate ratio of those that left vs stayed
        # TODO: Use something other than iterrows for efficiency
        all_row_ratios = []
        for _, row in df_combined.iterrows():
            ratio_left = row['Left']/row['All']
            ratio_stayed = row['Stayed']/row['All']
            all_row_ratios.append([ratio_stayed.round(2), ratio_left.round(2)])

        ratios_dataframe = pd.DataFrame(all_row_ratios)
        print(f"Ratios: {ratios_dataframe}\n")

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
        plt.savefig("Figures/group_box_plot.png")
        plt.show()
