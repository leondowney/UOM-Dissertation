import pandas as pd
from sklearn.utils import Bunch
import numpy as np


def load_pima():
    """
    Load the Pima Indians diabetes dataset and convert it to a sklearn.utils.Bunch object.

    :return Bunch: A Bunch object containing data, target, description, feature names, and target names.
    """
    df = pd.read_csv("./Pima Indians.csv")

    # Create a Bunch object and populate it with data
    sk_pima = Bunch()
    sk_pima.data = _get_data(df)
    sk_pima.target = _get_target(df)
    sk_pima.DESCR = _get_descr("Pima Indians", df)
    sk_pima.feature_names = _get_feature_names(df)
    sk_pima.target_names = _get_target_names(df)

    return sk_pima


def load_heart():
    """
    Load the heart diseases dataset and convert it to a sklearn.utils.Bunch object.

    :return Bunch: A Bunch object containing data, target, description, feature names, and target names.
    """
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
               "thal", "target"]
    df = pd.read_csv("./processed.cleveland.data", names=columns)

    # Create a Bunch object and populate it with data
    sk_heart = Bunch()
    sk_heart.data = _get_data(df)
    sk_heart.target = _get_target(df)
    sk_heart.DESCR = _get_descr("heart diseases", df)
    sk_heart.feature_names = _get_feature_names(df)
    sk_heart.target_names = _get_target_names(df)

    return sk_heart


def _get_data(df):
    """
    Extract feature data from the DataFrame.

    :param df: df (DataFrame): DataFrame containing the data.
    :return: numpy.ndarray: NumPy array containing feature data.
    """

    data_r = df.iloc[:, 0:len(df.columns) - 1]
    data_np = np.array(data_r, dtype=np.float64)
    return data_np


def _get_target(df):
    """
    Extract target labels from the dataset.
    :param df: df (DataFrame): DataFrame containing the data.
    :return: numpy.ndarray: NumPy array containing target labels.
    """
    target_r = df.iloc[:, len(df.columns) - 1]
    target_np = np.array(target_r, dtype=np.int32)
    return target_np


def _get_descr(dataset_name, df):
    """
    Generate a description of the dataset.
    :param dataset_name: The name of the dataset.
    :param df: DataFrame containing the data.
    :return: A string describing the dataset.
    """
    text = "{} dataset, number of samples: {};".format(dataset_name, len(df))
    return text


def _get_feature_names(df):
    """
    :param df: df (DataFrame): DataFrame containing the data.
    :return: list: A list of feature names.
    """

    feature_names = df.columns[:-1].tolist()
    return feature_names


def _get_target_names(df):
    """
    :param df: df (DataFrame): DataFrame containing the data.
    :return: list: A list of target names.
    """
    target_names = [df.columns[-1]]
    return target_names
