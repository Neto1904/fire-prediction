import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier

import warnings
warnings.filterwarnings(action='ignore')

def main():
    data = pd.read_csv('data/forestfires.csv')
    X_train, X_test, y_train, y_test = process_inputs(data)

    apply_mlp_classifier(X_train, y_train, X_test, y_test, (16, 16))
    apply_linear_classifier(X_train, y_train, X_test, y_test)
    apply_k_means_clustering(X_train, X_test, y_test)

def apply_mlp_classifier(X_train, y_train, X_test, y_test, hidden_layer_sizes):
    mlp_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=400)
    mlp_classifier.fit(X_train, y_train)
    score = mlp_classifier.score(X_test, y_test)
    print(score)

def apply_linear_classifier(X_train, y_train, X_test, y_test):
    linear_classifier = LogisticRegression()
    linear_classifier.fit(X_train, y_train)
    score = linear_classifier.score(X_test, y_test)
    print(score)

def apply_k_means_clustering(X_train, X_test, y_test):
    k_means_clustering = KMeans()
    k_means_clustering.fit(X_train)
    for X, y in zip(X_test, y_test):
        output = k_means_clustering.predict(X)
        print(output, y)

def process_inputs(data):
    df = data.copy()
    df = ordinal_encode(df)

    y = df['area'].apply(lambda x: 1 if x > 0 else 0)
    X = df.drop('area', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    
    transformer = get_transformer('scaler')
    transformer.fit(X_train)

    X_train = pd.DataFrame(transformer.transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(transformer.transform(X_test), columns=X.columns)
    return X_train, X_test, y_train, y_test

def get_transformer(transform_type):
    transformers = {
        'normalizer': Normalizer(),
        'scaler': StandardScaler()
    }
    return transformers[transform_type]

def ordinal_encode(df):
    df = df.copy()
    month_order = [
            'jan',
            'feb',
            'mar',
            'apr',
            'may',
            'jun',
            'jul',
            'aug',
            'sep',
            'oct',
            'nov',
            'dec'
        ]
    day_order = [
            'sun',
            'mon',
            'tue',
            'wed',
            'thu',
            'fri',
            'sat'
        ]
    df['month'] = df['month'].apply(lambda x: month_order.index(x))
    df['day'] = df['day'].apply(lambda x: day_order.index(x))
    return df

main()