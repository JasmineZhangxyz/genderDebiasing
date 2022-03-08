import pandas as pd
import urllib.request as requests


def get_urls():
    """
    NOTE: this function will not work if the csv files being used are moved outside of the folder that this file is in!
    Output: pds (treated as a list) of urls for test, train, and validation.
    """
    # test
    test_filepath = "gutenberg-test-urls.csv"
    test_data = pd.read_csv(test_filepath)
    test_urls = test_data["url"]

    # train
    train_filepath = "gutenberg-train-urls.csv"
    train_data = pd.read_csv(train_filepath)
    train_urls = train_data["url"]

    # validation
    valid_filepath = "gutenberg-validation-urls.csv"
    valid_data = pd.read_csv(valid_filepath)
    valid_urls = valid_data["url"]

    return test_urls, train_urls, valid_urls


def get_data(test_urls, train_urls, valid_urls):
    """
    Input: pd (treated as a list) of urls for test, train, and validation.
    Output: 3 lists of (unprocessed) words that appear in test, train, and validation data sets
    """
    def read_urls(urls):
        data = []
        for link in urls:
            temp_data = requests.urlopen(link)
            words = temp_data.read()
            data.append(words)
        return data

    test_data = read_urls(test_urls)
    train_data = read_urls(train_urls)
    valid_data = read_urls(valid_urls)

    return test_data, train_data, valid_data
