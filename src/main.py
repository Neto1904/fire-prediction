import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings(action='ignore')

def main():
    data = pd.read_csv('data/forestfires.csv')
    X_train, X_test, y_train, y_test = process_inputs(data)

    # apply_mlp_classifier(X_train, y_train, X_test, y_test, (8, 8, 8))
    # apply_mlp_classifier(X_train, y_train, X_test, y_test, (8, 8))
    # apply_mlp_classifier(X_train, y_train, X_test, y_test, (16, 16))
    # apply_dtree_classifier(X_train, y_train, X_test, y_test)
    apply_k_means_clustering(X_train, X_test, y_test)

def apply_mlp_classifier(X_train, y_train, X_test, y_test, hidden_layer_sizes):
    mlp_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=400)
    mlp_classifier.fit(X_train, y_train)
    y_pred = mlp_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    plt.figure()
    plot_confusion_matrix(mlp_classifier, X_test, y_test)
    plt.title(f'Confusion matrix - {hidden_layer_sizes}')
    plt.savefig(f'plots/confusion_matrix - {hidden_layer_sizes}.jpg', format='jpg')
    print('MLP Accuracy:', accuracy)
    print('MLP F1 score:', f1)

def apply_dtree_classifier(X_train, y_train, X_test, y_test):
    dtree_classifier = DecisionTreeClassifier()
    dtree_classifier.fit(X_train, y_train)
    y_pred = dtree_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    plt.figure()
    plot_confusion_matrix(dtree_classifier, X_test, y_test)
    plt.title(f'Confusion matrix - Decision Tree')
    plt.savefig(f'plots/confusion_matrix - Decision Tree.jpg', format='jpg')
    print('DTree Accuracy:', accuracy)
    print('DTree F1 score:', f1)

def apply_k_means_clustering(X_train, X_test, y_test):
    k_means_clustering = KMeans(n_clusters=2)
    k_means_clustering.fit(X_train)
    correct_predictions = 0
    y_test = np.array(y_test.values)
    predictions = []
    for y_pred, y_true in zip(k_means_clustering.predict(X_test), y_test):
        if(y_true == y_pred): correct_predictions += 1
        predictions.append(y_pred)
    f1 = f1_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion matrix - K-means')
    plt.savefig(f'plots/confusion_matrix - K-means.jpg', format='jpg')
    print('K-means Accuracy:' ,correct_predictions/len(X_test))
    print('K-means F1 score:', f1)

def process_inputs(data):
    df = data.copy()
    df = ordinal_encode(df)

    y = df['area'].apply(lambda x: 1 if x > 0 else 0)
    X = df.drop(['area', 'FFMC', 'DMC', 'DC', 'ISI'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
   
    transformer = get_transformer('normalizer')
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