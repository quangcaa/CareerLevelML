import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTEN


def filter_location(location):
    result = re.findall(r",\s[A-Z]{2}$", location)
    if len(result) != 0:
        return result[0][2:]
    else:
        return location


def load_and_preprocess_data(file_path, target_column="career_level", test_size=0.2, random_state=142):
    data = pd.read_excel(file_path, engine="odf", dtype=str)
    data = data.dropna(axis=0)
    data["location"] = data["location"].apply(filter_location)

    x = data.drop(target_column, axis=1)
    y = data[target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return x_train, x_test, y_train, y_test


def create_preprocessor():
    return ColumnTransformer(transformers=[
        ("title_ft", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "title"),
        ("location_ft", OneHotEncoder(handle_unknown="ignore"), ["location"]),
        ("desc_ft", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.95), "description"),
        ("function_ft", OneHotEncoder(handle_unknown="ignore"), ["function"]),
        ("industry_ft", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "industry"),
    ])


def balance_dataset(x_train, y_train, sampling_strategy=None):
    if sampling_strategy is None:
        sampling_strategy = {
            "bereichsleiter": 1000,
            "director_business_unit_leader": 500,
            "specialist": 450,
            "managing_director_small_medium_company": 400
        }

    ros = SMOTEN(random_state=0, k_neighbors=2, sampling_strategy=sampling_strategy)
    x_train_balanced, y_train_balanced = ros.fit_resample(x_train, y_train)

    return x_train_balanced, y_train_balanced
