import pandas as pd
import sys

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load data from CSV files and merge into a single dataframe
    :param messages_filepath:
    :param categories_filepath:
    :return:
    """

    # Load messages csv
    messages = pd.read_csv(messages_filepath)

    # Load categories csv
    categories = pd.read_csv(categories_filepath)

    # Merge data sets
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    """
    Split categories column and drop duplicate rows from dataframe
    :param df:
    :return:
    """

    # Split categories column
    df = split_categories(df)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    return df


def split_categories(df):
    """
    Split "categories" column into dedicated columns based on the leading label string
    :param df:
    :return:
    """

    # Split categories column
    split_df = df["categories"].str.split(';', expand=True)

    # Retrieve column names
    row = split_df.iloc[0]
    cat_cols = [x.replace('-0', '').replace('-1', '') for x in row.tolist()]

    # Rename columns
    split_df.columns = cat_cols

    # Convert columns to numeric only
    for column in cat_cols:

        # Retrieve last character and set to numeric
        split_df[column] = split_df[column].str[-1].astype(int)

        # Ensure number is no greater than 1
        split_df[column] = split_df[column].apply(lambda x: 1 if x > 1 else x)

    # Drop categories column from original dataframe
    df.drop(columns=['categories'], inplace=True)

    # Concatenate dataframes
    comb_df = pd.concat([df, split_df], axis=1)

    return comb_df


def save_data(df, database_filename):
    """
    Connect to SQLite database and send dataframe to a table
    :param df:
    :param database_filename:
    :return:
    """

    engine = create_engine(database_filename)

    df.to_sql('messages', engine, index=False)


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
