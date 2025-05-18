import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler, SMOTEN
from sklearn.model_selection import GridSearchCV


def filter_location(location):
    result = re.findall(r",\s[A-Z]{2}$", location)
    if len(result) != 0:
        return result[0][2:]
    else:
        return location


data = pd.read_excel("data/final_project.ods", engine="odf", dtype=str)
data = data.dropna(axis=0)
data["location"] = data["location"].apply(filter_location)

target = "career_level"

x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=142, stratify=y)

ros = SMOTEN(random_state=0, k_neighbors=2, sampling_strategy={
    "bereichsleiter": 1000,
    "director_business_unit_leader": 500,
    "specialist": 450,
    "managing_director_small_medium_company": 400
})
x_train, y_train = ros.fit_resample(x_train, y_train)

# vectorizer = TfidfVectorizer(stop_words="english")
# result = vectorizer.fit_transform(x_train["title"])
# result = pd.DataFrame(result.todense())
# print(len(vectorizer.vocabulary_))
# print(result.shape)

preprocessor = ColumnTransformer(transformers=[
    ("title_ft", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "title"),
    ("location_ft", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("desc_ft", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.95), "description"),
    ("function_ft", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry_ft", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "industry"),
])

cls = Pipeline(steps=[
    ("preprocessor", preprocessor),  # (6458, 8xxx)
    ("features_selector", SelectPercentile(chi2, percentile=5)),
    ("model", RandomForestClassifier())
])

params = {
    # "model__n_estimators": [50, 100, 200],
    "model__criterion": ["gini", "entropy", "log_loss"],
    "features_selector__percentile": [1, 5, 10]
}
grid_search = GridSearchCV(estimator=cls, param_grid=params, scoring="recall_weighted", cv=4, verbose=2)
grid_search.fit(x_train, y_train)
y_predicted = grid_search.predict(x_test)
print(classification_report(y_test, y_predicted))
