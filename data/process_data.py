import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
      Function:
      Load data from two csv file and then merge them
      Args:
      messages_filepath (str): the file path of messages csv file
      categories_filepath (str): the file path of categories csv file
      Return:
      df (DataFrame): A dataframe of messages and categories
      """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id', how='left')
    return df


def clean_data(df):
    """
      Function:
      Clean the df
      Args:
      df (DataFrame): A dataframe of messages and categories needs to be cleaned
      Return:
      df (DataFrame): A cleaned dataframe of messages and categories
      """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=";", expand=True)
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = (row.str.slice(0, -2).apply(lambda x: x))
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.slice(-1).apply(lambda x: x)
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from df
    df = df.drop(['categories'], axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    # drop values with 2 in related column
    df = df.drop(df[df['related'] == 2].index)
    return df


def save_data(df, database_filename):
  """
     Function:
     Save the Dataframe in the database
     Args:
     df (DataFrame): A dataframe of messages and categories
     database_filename (str): The file name of the database
  """
  engine = create_engine('sqlite:///'+database_filename)
  df.to_sql('disaster_response', engine, index=False, if_exists='replace')
  


def main():
  if len(sys.argv) == 4:

    messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)

    print('Cleaned data saved to database!')

  else:
    print('Please provide the filepaths of the messages and categories '
          'datasets as the first and second argument respectively, as '
          'well as the filepath of the database to save the cleaned data '
          'to as the third argument. \n\nExample: python process_data.py '
          'disaster_messages.csv disaster_categories.csv '
          'DisasterResponse.db')


if __name__ == '__main__':
  main()
