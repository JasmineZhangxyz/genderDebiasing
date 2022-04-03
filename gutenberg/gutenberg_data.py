import pandas as pd
import urllib.request as requests


def get_urls(data_type):
    """
    NOTE: this function will not work if the csv files being used are moved outside of the folder that this file is in!
    Output: pds (treated as a list) of urls for test, train, and validation.
    """
    
    # test
    if (data_type == 'test'):
        test_filepath = "gutenberg/gutenberg-test-urls.csv"
        test_data = pd.read_csv(test_filepath)
        test_urls = test_data["url"]
        return test_urls

    # train
    if (data_type == 'train'):
        train_filepath = "gutenberg/gutenberg-train-urls.csv"
        train_data = pd.read_csv(train_filepath)
        train_urls = train_data["url"]
        return train_urls

    # validation
    if (data_type == 'valid'):
        valid_filepath = "gutenberg/gutenberg-validation-urls.csv"
        valid_data = pd.read_csv(valid_filepath)
        valid_urls = valid_data["url"]
        return valid_urls

def read_data_from_urls(urls):
    data = []
    for link in urls:
        temp_data = requests.urlopen(link)
        words = temp_data.read()
        data.append(words)
    return data

def get_data(test_urls, train_urls, valid_urls):
    """
    Input: pd (treated as a list) of urls for test, train, and validation.
    Output: 3 lists of (unprocessed) words that appear in test, train, and validation data sets
    """
    
    test_data = read_data_from_urls(test_urls)
    train_data = read_data_from_urls(train_urls)
    valid_data = read_data_from_urls(valid_urls)

    return test_data, train_data, valid_data
