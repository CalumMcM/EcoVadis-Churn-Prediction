"""
This module contains the DataLoader class which allows a user to load in and
clean an excel spreadsheet of data as well as apply sentiment analysis to
certain classes. 
"""
import ssl

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import preprocessing


class DataLoader():
    """
    The DataLoader will load an excel spreadsheet from a given directory and
    clean the data ensuring that there are no NaN values present. The DataLoader
    can also be used to apply label encoding or sentiment analysis to certain
    fields.

    Attributes:
        * self.dataset_dir: (string)

    Methods:
        * load_and_clean
        * apply_label_encoding
        * apply_sentiment_analysis
    """

    def __init__(self, dataset_dir: str) -> None:
        """
        Init function for the DataLoader class 

        Inputs:
            * dataset_dir: (string) Directory location for the dataset
        """
        self.dataset_dir = dataset_dir

    def load_and_clean(self) -> pd.DataFrame:
        """
        Loads an excel spreadsheet from the global excel directory

        Returns:
            * clean_dataframe: (pd.DataFrame) Pandas dataframe of the given
              excel spreadsheet that has had all NaN values errors replaced.
        """
        excel_dataframe = pd.read_excel(self.dataset_dir)

        # Replaces all NaN values in an array with empty strings. This is okay
        # for this use case where the only NaN values are in columns of type
        # string. However a more robust solution would check the column type
        # first then replace accordingly.
        clean_dataframe = excel_dataframe.fillna("")

        return clean_dataframe

    def apply_label_encoding(self, df: pd.DataFrame,
                             cols_to_apply: list[str]) -> pd.DataFrame:
        """
        Takes a dataframe and a list of column headers that should be encoded
        from their string values to integers, e.g., ["France", "Germany"]
        becomes [0,  1]

        Inputs:
            * df: (pd.DataFrame) The dataframe containing the data
            * cols_to_apply: (list(string)) The columns to apply label encoding
              to

        Returns:
            * df: (pd.DataFrame) The original dataframe but the columns
              contained in the cols_to_apply input have been converted from
              strings to integer encodings
        """
        encoder = preprocessing.LabelEncoder()

        # Create a dataframe that just contains the columns of interest
        cols_of_interest = df[cols_to_apply]

        # Apply label encoding
        cols_of_interest = cols_of_interest.apply(encoder.fit_transform)

        # Merge the updated columns of interest back to the original dataset
        df[cols_to_apply] = cols_of_interest

        return df

    def apply_sentiment_analysis(self, df: pd.DataFrame,
                                 col_name: str) -> tuple[pd.DataFrame, str]:
        """
        Applies a VADER sentiment analyser to the given column. The compound
        score reported by VADER is interpreted as:
            * > 0.05 = Review is positive
            * 0.05 < && < 0.05 = Review is neutral
            * < 0.05 = Review is negative

        Inputs:
            * df: (pd.DataFrame) The dataframe containing the data
            * col_name: (string) Column which sentiment analysis will be applied
              to. 

        Returns:
            * df: (pd.DataFrame) The same dataframe that was passed into the
              model but the col_name column has had sentiment analysis applied.
        """
        # Download the Vader Lexicon
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download('vader_lexicon')

        # Create VADER sentiment analyser
        analyzer = SentimentIntensityAnalyzer()

        # Extract column to apply sentiment analysis to
        col_to_apply = df[col_name].to_numpy()

        sentiments = np.zeros(len(col_to_apply))

        # Apply sentiment analysis to each entry in the columns
        for idx, text in enumerate(col_to_apply):
            # Get the compound score and sort the combined
            # probability into one of three columns
            compound_score = analyzer.polarity_scores(text[0])["compound"]

            if compound_score > 0.05:
                sentiments[idx] = 1
            elif compound_score < -0.05:
                sentiments[idx] = -1
            else:
                sentiments[idx] = 0

        # Change to format of Pandas DataFrame and overwrite original customer
        # feedback
        sentiments = np.transpose([sentiments])

        df[col_name] = sentiments

        return df
