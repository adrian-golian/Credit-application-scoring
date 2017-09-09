import sys
from csv import reader
from nltk import word_tokenize as tokenize
from pandas import read_csv
import datetime
from collections import Counter
import string
import time
import random
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import resample

from matplotlib import pyplot as plt

def train_evaluate_model(model, X_train, X_test, y_train, y_test, class_labels, class_frequency, model_name):
    # Train the model
    model.fit(X_train, y_train)

    # Now we can use the model to predict classifications for our test features.
    predictions = model.predict(X_test)

    if hasattr(model, 'predict_proba'):
        predictions_proba = model.predict_proba(X_test)
        sorted_labels = sorted(list(enumerate(y_test)), key=lambda tup: predictions_proba[tup[0]][1], reverse=True)
        sorted_labels = map(lambda tup: tup[1], sorted_labels)
        deciles = np.array_split(sorted_labels, 10)
        rank_ordering = [sum(decile)/float(len(decile)) for decile in deciles]
        print "rank_ordering", rank_ordering
        plt.plot(list(reversed(range(1,11))), rank_ordering)
        plt.xlabel('Deciles of customers')
        plt.ylabel('Bad rate')
        plt.gca().invert_xaxis()
        plt.title("Rank ordering - {}".format(model_name))
        plt.savefig("rank_ordering_{}".format(model_name.replace(" ", "_").lower()))
        plt.show()

    if hasattr(model, "feature_importances_"):
        print "Feature importances", model.feature_importances_

    # Calculate precision and recall
    precision_scores_list = metrics.precision_score(y_test, predictions, average=None, labels=class_labels)
    recall_scores_list = metrics.recall_score(y_test, predictions, average=None, labels=class_labels)

    prec_recall_fscore = metrics.precision_recall_fscore_support(y_test, predictions)
    report = metrics.classification_report(y_test, predictions)

    print prec_recall_fscore
    print report

    precision_scores = dict(zip(class_labels, precision_scores_list))
    recall_scores = dict(zip(class_labels, recall_scores_list))

    print("Average precision=%.3f%%" %(100*metrics.precision_score(y_test, predictions, average='weighted')))
    print("Average recall=%.3f%%" %(100*metrics.recall_score(y_test, predictions, average='weighted')))
    print(metrics.confusion_matrix(y_test, predictions))
    print("Gini: {}".format(2*metrics.roc_auc_score(y_test, predictions) - 1))

    return model


def vectorise(df, fit=True, vectorisers=None):
    if vectorisers is None:
        vectorisers = {}
    for column in df:
        if fit:
            # vect = CountVectorizer()
            vect = LabelEncoder()
            try:
                df[column] = vect.fit_transform(df[column])
            except ValueError:
                df.drop(column, inplace=True, axis=1)
                continue
            vectorisers[column] = vect
        else:
            if column not in vectorisers:
                df.drop(column, inplace=True, axis=1)
                continue
            vect = vectorisers[column]
            df[column] = vect.fit_transform(df[column])
    return df, vectorisers

def listify_vector(vec):
    if isinstance(vec, list):
        return vec
    else:
        return vec.tolist()

# filename = "data/Twitter-hate_speech-labeled_data.csv"

account_train_filename = "test_data/raw_account_70_new.csv" 
account_test_filename = "test_data/raw_account_30_new.csv"

train_filename = "test_data/raw_data_70_new.csv" 
test_filename = "test_data/raw_data_30_new.csv"

# train_filename = "test_data/raw_enquiry_70_new.csv" 
# test_filename = "test_data/raw_enquiry_30_new.csv"

target_label = "Bad_label"


training_df = read_csv(train_filename)
test_df = read_csv(test_filename)

col_list = training_df.columns.tolist()


account_train_df = read_csv(account_train_filename)
account_test_df = read_csv(account_test_filename)

convert_date_to_timestamp = lambda text: time.mktime(datetime.datetime.strptime(text, "%d-%b-%y").timetuple())

# Dealing with nans
account_train_df = account_train_df[pd.notnull(account_train_df['last_paymt_dt'])]
account_train_df = account_train_df[pd.notnull(account_train_df['opened_dt'])]
account_train_df = account_train_df[pd.notnull(account_train_df["cur_balance_amt"])]
account_train_df = account_train_df[pd.notnull(account_train_df["creditlimit"])]
account_train_df = account_train_df[pd.notnull(account_train_df["cashlimit"])]

account_test_df = account_test_df[pd.notnull(account_test_df['last_paymt_dt'])]
account_test_df = account_test_df[pd.notnull(account_test_df['opened_dt'])]
account_test_df = account_test_df[pd.notnull(account_test_df["cur_balance_amt"])]
account_test_df = account_test_df[pd.notnull(account_test_df["creditlimit"])]
account_test_df = account_test_df[pd.notnull(account_test_df["cashlimit"])]

account_train_df[['opened_dt', 'last_paymt_dt', 'cur_balance_amt', 'creditlimit', 'cashlimit', 'customer_no']].dropna(axis=0, inplace=True)
account_train_df['total_diff_lastpaymt_opened_dt'] = (account_train_df["opened_dt"].apply(convert_date_to_timestamp) - account_train_df["last_paymt_dt"].apply(convert_date_to_timestamp)) / 86400.0

account_test_df[['opened_dt', 'last_paymt_dt', 'cur_balance_amt', 'creditlimit', 'cashlimit', 'customer_no']].dropna(axis=0, inplace=True)
account_test_df['total_diff_lastpaymt_opened_dt'] = (account_test_df["opened_dt"].apply(convert_date_to_timestamp) - account_test_df["last_paymt_dt"].apply(convert_date_to_timestamp)) / 86400.0

training_df = pd.merge(account_train_df.groupby(account_train_df['customer_no'], as_index=False).sum(), training_df[[target_label, 'customer_no']], on='customer_no')
training_df_means = account_train_df.groupby(account_train_df['customer_no'], as_index=False).mean()
test_df = pd.merge(account_test_df.groupby(account_test_df['customer_no'], as_index=False).sum(), test_df[[target_label, 'customer_no']], on='customer_no')
test_df_means = account_test_df.groupby(account_test_df['customer_no'], as_index=False).mean()

training_df['utilisation_trend'] = training_df['cur_balance_amt'] / training_df['creditlimit'] / training_df_means['cur_balance_amt'] * (training_df_means['creditlimit']+training_df_means['cashlimit'])
training_df['ratio_currbalance_creditlimit'] = training_df['cur_balance_amt'] / training_df['creditlimit']
training_df.drop(['cur_balance_amt', 'creditlimit', 'cashlimit'], axis=1, inplace=True)

test_df['utilisation_trend'] = test_df['cur_balance_amt'] / test_df['creditlimit'] / test_df_means['cur_balance_amt'] * (test_df_means['creditlimit']+test_df_means['cashlimit'])
test_df['ratio_currbalance_creditlimit'] = test_df['cur_balance_amt'] / test_df['creditlimit']
test_df.drop(['cur_balance_amt', 'creditlimit', 'cashlimit'], axis=1, inplace=True)

training_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
training_df.dropna(subset=['utilisation_trend', 'ratio_currbalance_creditlimit', 'actualpaymentamount'], axis=0, inplace=True)
test_df.dropna(subset=['utilisation_trend', 'ratio_currbalance_creditlimit', 'actualpaymentamount'], axis=0, inplace=True)

training_df['amt_past_due'].fillna(value=0, inplace=True)
test_df['amt_past_due'].fillna(value=0, inplace=True)

training_df.dropna(axis=0, inplace=True)
test_df.dropna(axis=0, inplace=True)


"""
The commented code below is what I used to prepare data
for model training, when using all* features from raw_data_70_new and raw_data_30_new
(* -> some features were removed because of very low support)
"""

# features_support = []
# for i in range(49):
#     features_support.append((i*500, len(training_df.dropna(axis=1, thresh=i*500).columns)))

# import matplotlib.pyplot as plt
# plt.scatter(*zip(*features_support))
# plt.title("Number of features vs no. of customers with each feature")
# plt.savefig("feature_support.png")


# import scipy.sparse as sp
# vect = CountVectorizer(ngram_range=(1, 1))

# # training_df = training_df.dropna(axis=1, thresh=20000)
# cols_to_drop = ["feature_75", "feature_70", "feature_63", "feature_21", "opened_dt", "entry_time", "feature_1", "feature_24", "feature_16", "feature_15", "feature_54", "feature_2"]
# cols_to_drop = [col for col in cols_to_drop if col in training_df.columns]
# training_df.drop(cols_to_drop, axis=1, inplace=True)
# test_df.drop(cols_to_drop, axis=1, inplace=True)


# NUMERICS = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# num_training_df = training_df.select_dtypes(include=NUMERICS).fillna(method='pad')
# nonnum_training_df = training_df.select_dtypes(exclude=NUMERICS).fillna(method='pad')
# nonnum_training_df, vectorisers = vectorise(nonnum_training_df)
# training_df = pd.concat([num_training_df, nonnum_training_df], axis=1)

# num_test_df = test_df.select_dtypes(include=NUMERICS).fillna(method='ffill')
# nonnum_test_df = test_df.select_dtypes(exclude=NUMERICS).fillna(method='ffill')#.apply(lambda col: vect.fit_transform(col))
# nonnum_test_df, vectorisers = vectorise(nonnum_test_df, fit=False, vectorisers=vectorisers)
# test_df = pd.concat([num_test_df, nonnum_test_df], axis=1)
# test_df.drop([col for col in test_df.columns if col not in training_df.columns], axis=1, inplace=True)



used_features = training_df.columns.tolist()
used_features.remove(target_label)

# Separate majority and minority classes
df_majority = training_df[training_df[target_label]==0]
df_minority = training_df[training_df[target_label]==1]

# # Downsample majority class
# df_majority_downsampled = resample(df_majority, 
#                                  replace=False,     # sample without replacement
#                                  n_samples=1004,     # to get a number closer to the minority size
#                                  random_state=1345678) # reproducible results
# training_df_downsampled = pd.concat([df_majority_downsampled, df_minority])

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=22892,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
training_df = pd.concat([df_majority, df_minority_upsampled])


print "Model features", training_df.drop(target_label, axis=1).columns
X_train = training_df.drop(target_label, axis=1).values.tolist()
X_test = test_df.drop(target_label, axis=1).values.tolist()
X_train = map(listify_vector, X_train)
X_test = map(listify_vector, X_test)

y_train = training_df[target_label].tolist()
y_test = test_df[target_label].tolist()

class_frequency = Counter(y_train)
class_labels = list(set(y_train))

# # Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=126452341)
# forest = RandomForestClassifier(n_estimators=250, random_state=0)

train_evaluate_model(forest, X_train, X_test, y_train, y_test, class_labels, class_frequency, "ExtraTreesClassifier")

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]


# Make random predictions for benchmark
rand_predictions = [class_labels[random.randint(0,1)] for label in y_test]

precision = metrics.precision_score(y_test, rand_predictions, average='weighted')
recall = metrics.recall_score(y_test, rand_predictions, average='weighted')

print("Predicting randomly")
print("Precision=%.3f%%" %(100*precision))
print("Recall=%.3f%%" %(100*recall))
print(metrics.confusion_matrix(y_test, rand_predictions))


# Make random predictions, exploiting class imbalances, for banchmark
rand_predictions = [random.choice(y_test) for label in y_test]

precision = metrics.precision_score(y_test, rand_predictions, average='weighted')
recall = metrics.recall_score(y_test, rand_predictions, average='weighted')

print("Predicting randomly (with existing class frequencies)")
print("Precision=%.3f%%" %(100*precision))
print("Recall=%.3f%%" %(100*recall))
print(metrics.confusion_matrix(y_test, rand_predictions))

# Predict most common class only, for banchmark
rand_predictions = [class_frequency.most_common(1)[0][0]] * len(y_test)

precision = metrics.precision_score(y_test, rand_predictions, average='weighted')
recall = metrics.recall_score(y_test, rand_predictions, average='weighted')

print("Predicting most common class")
print("Precision=%.3f%%" %(100*precision))
print("Recall=%.3f%%" %(100*recall))
print(metrics.confusion_matrix(y_test, rand_predictions))

models = [("Naive Bayes", naive_bayes.BernoulliNB),
        ("Decision tree", DecisionTreeClassifier),
        ("Random forest", lambda: RandomForestClassifier(n_estimators=250)),
        ("Linear SVC", LinearSVC), ("Logistic Regression", LogisticRegression)]

# Using the same train-test split, evaluate performance of several models
for model_name, model in models:
    print("\nTraining a %s" %model_name)
    train_evaluate_model(model(), X_train, X_test, y_train, y_test, class_labels, class_frequency, model_name)
