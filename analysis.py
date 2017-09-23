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

from xgboost import XGBClassifier
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
from sklearn.grid_search import GridSearchCV

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
        print("Gini: {}".format(2*metrics.roc_auc_score(y_test, map(lambda tup: tup[1], predictions_proba)) - 1))

    if hasattr(model, "feature_importances_"):
        print "Feature importances:", model.feature_importances_

    print(metrics.classification_report(y_test, predictions))
    print(metrics.confusion_matrix(y_test, predictions))

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

def convert_pay_his_to_int(pay_his_str):
    try:
        return int(pay_his_str)
    except:
        return 0

def process_payment_history(pay_his_str):
    pay_his_str = pay_his_str.strip('"""""""')
    pay_his = [convert_pay_his_to_int(pay_his_str[3*i:3*(i+1)]) for i in range(len(pay_his_str)/3)]
    pay_his_30dpd = [n > 30 for n in pay_his]
    return pay_his_30dpd

def get_months_last_30_plus(pay_his):
    for i in range(1,len(pay_his)-1):
        if pay_his[-i]:
            return 1.0/i
    else:
        return 0# np.inf

pad_zeros_left = lambda iterable, n: np.lib.pad(iterable, (n,0), 'constant', constant_values=(0,0))

def get_history_avg_dpd_0_29_bucket(pay_his_list):
    pay_his_list = [[int(value) for value in pay_his] for pay_his in pay_his_list]
    max_his_len = max([len(pay_his) for pay_his in pay_his_list])

    padded_pay_his_list = []
    for pay_his in pay_his_list:
        padded_pay_his_list.append(pad_zeros_left(pay_his, max_his_len - len(pay_his)))
    summed_pay_his =  map(sum, zip(*padded_pay_his_list))
    return sum(summed_pay_his)/float(max_his_len)

def find_min(iterable):
    int_iterable = [n for n in iterable if not np.isnan(n)]
    if len(int_iterable) == 0:
        return 0
    else:
        return min(int_iterable)

def get_percentage_unsecured(iterable):
    unsecured_types = [5, 6, 8, 9, 10, 12, 16, 35, 40, 41, 43, 80, 81, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 00, 98, 99]
    bool_iterable = [int(n in unsecured_types) for n in iterable]
    return sum(bool_iterable) / float(len(bool_iterable))

account_train_filename = "test_data/raw_account_70_new.csv" 
account_test_filename = "test_data/raw_account_30_new.csv"

train_filename = "test_data/raw_data_70_new.csv" 
test_filename = "test_data/raw_data_30_new.csv"

enquiry_train_filename = "test_data/raw_enquiry_70_new.csv" 
enquiry_test_filename = "test_data/raw_enquiry_30_new.csv"

target_label = "Bad_label"

features_to_use = ['payment_history_avg_dpd_0_29_bucket', 'total_diff_lastpaymt_opened_dt',
'min_months_last_30_plus', 'utilisation_trend',
'count_enquiry_recency_365', 'ratio_currbalance_creditlimit',
'mean_diff_lastpaymt_opened_dt', 'mean_diff_open_enquiry_dt',   
'payment_history_mean_length', 'max_freq_enquiry',
'count_enquiry_recency_90', 'perc_unsecured_others']


training_df = read_csv(train_filename)
test_df = read_csv(test_filename)

account_train_df = read_csv(account_train_filename)
account_test_df = read_csv(account_test_filename)

enquiry_train_df = read_csv(enquiry_train_filename)
enquiry_test_df = read_csv(enquiry_test_filename)

convert_date_to_timestamp = lambda text: time.mktime(datetime.datetime.strptime(text, "%d-%b-%y").timetuple())

# Dealing with nans
enquiry_train_df = enquiry_train_df[pd.notnull(enquiry_train_df['enquiry_dt'])]
enquiry_train_df = enquiry_train_df[pd.notnull(enquiry_train_df['dt_opened'])]
enquiry_train_df = enquiry_train_df[pd.notnull(enquiry_train_df['enq_purpose'])]
enquiry_test_df = enquiry_test_df[pd.notnull(enquiry_test_df['enquiry_dt'])]
enquiry_test_df = enquiry_test_df[pd.notnull(enquiry_test_df['dt_opened'])]
enquiry_test_df = enquiry_test_df[pd.notnull(enquiry_test_df['enq_purpose'])]

account_train_df = account_train_df[pd.notnull(account_train_df['last_paymt_dt'])]
account_train_df = account_train_df[pd.notnull(account_train_df['opened_dt'])]
account_train_df = account_train_df[pd.notnull(account_train_df["cur_balance_amt"])]
account_train_df = account_train_df[pd.notnull(account_train_df["creditlimit"])]
account_train_df = account_train_df[pd.notnull(account_train_df["cashlimit"])]
account_train_df = account_train_df[pd.notnull(account_train_df["paymenthistory1"])]

account_test_df = account_test_df[pd.notnull(account_test_df['last_paymt_dt'])]
account_test_df = account_test_df[pd.notnull(account_test_df['opened_dt'])]
account_test_df = account_test_df[pd.notnull(account_test_df["cur_balance_amt"])]
account_test_df = account_test_df[pd.notnull(account_test_df["creditlimit"])]
account_test_df = account_test_df[pd.notnull(account_test_df["cashlimit"])]
account_test_df = account_test_df[pd.notnull(account_test_df["paymenthistory1"])]

# Feature bulding:
account_train_df["paymenthistory1"] = account_train_df["paymenthistory1"].apply(process_payment_history)
account_test_df["paymenthistory1"] = account_test_df["paymenthistory1"].apply(process_payment_history)

#account_train_df[['opened_dt', 'last_paymt_dt', 'cur_balance_amt', 'creditlimit', 'cashlimit', 'customer_no']].dropna(axis=0, inplace=True)
account_train_df['total_diff_lastpaymt_opened_dt'] = (account_train_df["opened_dt"].apply(convert_date_to_timestamp) - account_train_df["last_paymt_dt"].apply(convert_date_to_timestamp)) / 86400.0

#account_test_df[['opened_dt', 'last_paymt_dt', 'cur_balance_amt', 'creditlimit', 'cashlimit', 'customer_no']].dropna(axis=0, inplace=True)
account_test_df['total_diff_lastpaymt_opened_dt'] = (account_test_df["opened_dt"].apply(convert_date_to_timestamp) - account_test_df["last_paymt_dt"].apply(convert_date_to_timestamp)) / 86400.0

training_df = pd.merge(account_train_df.groupby(account_train_df['customer_no'], as_index=False).sum(), training_df[[target_label, 'customer_no']], on='customer_no')
training_df_means = account_train_df.groupby(account_train_df['customer_no'], as_index=False).mean()
test_df = pd.merge(account_test_df.groupby(account_test_df['customer_no'], as_index=False).sum(), test_df[[target_label, 'customer_no']], on='customer_no')
test_df_means = account_test_df.groupby(account_test_df['customer_no'], as_index=False).mean()

temp_enquiry_train_df = enquiry_train_df['enquiry_dt'].apply(convert_date_to_timestamp)
training_df['count_enquiry_recency_365'] = temp_enquiry_train_df.apply(lambda x: int(x>1420070400)).groupby(enquiry_train_df['customer_no']).aggregate(lambda x: sum(x))
training_df['count_enquiry_recency_90'] = temp_enquiry_train_df.apply(lambda x: int(x>1443830400)).groupby(enquiry_train_df['customer_no']).aggregate(lambda x: sum(x))
temp_enquiry_test_df = enquiry_test_df['enquiry_dt'].apply(convert_date_to_timestamp)
test_df['count_enquiry_recency_365'] = temp_enquiry_test_df.apply(lambda x: int(x>1420070400)).groupby(enquiry_test_df['customer_no']).aggregate(lambda x: sum(x))
test_df['count_enquiry_recency_90'] = temp_enquiry_test_df.apply(lambda x: int(x>1443830400)).groupby(enquiry_test_df['customer_no']).aggregate(lambda x: sum(x))

temp_enquiry_train_df = enquiry_train_df['dt_opened'].apply(convert_date_to_timestamp) - enquiry_train_df['enquiry_dt'].apply(convert_date_to_timestamp)
training_df['mean_diff_open_enquiry_dt'] = temp_enquiry_train_df.groupby(enquiry_train_df['customer_no']).mean()
temp_enquiry_test_df = enquiry_test_df['dt_opened'].apply(convert_date_to_timestamp) - enquiry_test_df['enquiry_dt'].apply(convert_date_to_timestamp)
test_df['mean_diff_open_enquiry_dt'] = temp_enquiry_test_df.groupby(enquiry_test_df['customer_no']).mean()

temp_enquiry_train_df = enquiry_train_df.groupby(enquiry_train_df['customer_no']).aggregate(lambda x: list(x))
training_df['max_freq_enquiry'] = temp_enquiry_train_df['enq_purpose'].apply(lambda x: Counter(x).most_common(1)[0][0])
temp_enquiry_test_df = enquiry_test_df.groupby(enquiry_test_df['customer_no']).aggregate(lambda x: list(x))
test_df['max_freq_enquiry'] = temp_enquiry_test_df['enq_purpose'].apply(lambda x: Counter(x).most_common(1)[0][0])
training_df['perc_unsecured_others'] = temp_enquiry_train_df['enq_purpose'].apply(get_percentage_unsecured)
test_df['perc_unsecured_others'] = temp_enquiry_test_df['enq_purpose'].apply(get_percentage_unsecured)

temp_account_train_df = account_train_df['paymenthistory1'].apply(get_months_last_30_plus)
temp_account_train_df = temp_account_train_df.groupby(account_train_df['customer_no']).aggregate(lambda x: find_min(x))
training_df['min_months_last_30_plus'] = temp_account_train_df
temp_account_test_df = account_test_df['paymenthistory1'].apply(get_months_last_30_plus)
temp_account_test_df = temp_account_test_df.groupby(account_test_df['customer_no']).aggregate(lambda x: find_min(x))
test_df['min_months_last_30_plus'] = temp_account_test_df

temp_account_train_df = account_train_df[['paymenthistory1', 'customer_no']]
temp_account_train_df = temp_account_train_df.groupby(account_train_df['customer_no']).aggregate(lambda x: sum([len(e) for e in x]) / float(len(x)))
training_df['payment_history_mean_length'] = temp_account_train_df['paymenthistory1']
temp_account_test_df = account_test_df[['paymenthistory1', 'customer_no']]
temp_account_test_df = temp_account_test_df.groupby(account_test_df['customer_no']).aggregate(lambda x: sum([len(e) for e in x]) / float(len(x)))
test_df['payment_history_mean_length'] = temp_account_test_df['paymenthistory1']

temp_account_train_df = account_train_df['paymenthistory1'].groupby(account_train_df['customer_no']).aggregate(lambda x: get_history_avg_dpd_0_29_bucket(x))
training_df['payment_history_avg_dpd_0_29_bucket'] = temp_account_train_df
temp_account_test_df = account_test_df['paymenthistory1'].groupby(account_test_df['customer_no']).aggregate(lambda x: get_history_avg_dpd_0_29_bucket(x))
test_df['payment_history_avg_dpd_0_29_bucket'] = temp_account_test_df

training_df['mean_diff_lastpaymt_opened_dt'] = training_df_means['total_diff_lastpaymt_opened_dt']
test_df['mean_diff_lastpaymt_opened_dt'] = test_df_means['total_diff_lastpaymt_opened_dt']

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
training_df['actualpaymentamount'].fillna(value=0, inplace=True)
test_df['actualpaymentamount'].fillna(value=0, inplace=True)
training_df['paymentfrequency'].fillna(value=0, inplace=True)
test_df['paymentfrequency'].fillna(value=0, inplace=True)

training_df.dropna(axis=0, inplace=True)
test_df.dropna(axis=0, inplace=True)

training_df = training_df[features_to_use+[target_label]]
test_df = test_df[features_to_use+[target_label]]

used_features = training_df.columns.tolist()
used_features.remove(target_label)

# Separate majority and minority classes
df_majority = training_df[training_df[target_label]==0]
df_minority = training_df[training_df[target_label]==1]

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
training_df = pd.concat([df_majority, df_minority_upsampled])


print "Model features:", training_df.drop(target_label, axis=1).columns
X_train = training_df.drop(target_label, axis=1).values.tolist()
X_test = test_df.drop(target_label, axis=1).values.tolist()
X_train = np.array(map(listify_vector, X_train))
X_test = np.array(map(listify_vector, X_test))

y_train = training_df[target_label].tolist()
y_test = test_df[target_label].tolist()

class_frequency = Counter(y_train)
class_labels = list(set(y_train))

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

random_state = 123
models = [("Naive Bayes", naive_bayes.BernoulliNB),
        ("Decision tree", lambda: DecisionTreeClassifier(class_weight='balanced', random_state=random_state)),
        ("Extra Trees", lambda: ExtraTreesClassifier(n_estimators=30, max_depth=2, max_features=3,
                              random_state=random_state)),
        ("Random forest", lambda: RandomForestClassifier(n_estimators=250, max_depth=2, max_features=1, min_samples_split=90, min_samples_leaf=11, class_weight='balanced', random_state=random_state)),
        ("XGBoost", lambda: XGBClassifier(n_estimators=105, max_depth=2, learning_rate=0.01, seed=random_state, nthread=4)),
        ("Linear SVC", LinearSVC), ("Logistic Regression", lambda: LogisticRegression(class_weight='balanced'))]

# Using the same train-test split, evaluate performance of several models
for model_name, model in models:
    print("\nTraining a %s" %model_name)
    tuned_parameters = {
                        "Random forest" : [{'n_estimators': [250],
                                     'max_depth': [2], 'max_features': [1],
                                     'min_samples_leaf': [11],
                                     'min_samples_split': [70, 80, 90]}],
                        "XGBoost" : [{'n_estimators': [105],
                                     'max_depth': [2], 'learning_rate':[0.01],
                                     'subsample': [0.75]
                                    }]
                        }
    do_grid_search = False
    if do_grid_search:
        grid = GridSearchCV(model(), tuned_parameters[model_name], cv=5)
        train_evaluate_model(grid, X_train, X_test, y_train, y_test, class_labels, class_frequency, model_name)
        for params, mean_score, scores in grid.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print grid.best_params_
    else:
        train_evaluate_model(model(), X_train, X_test, y_train, y_test, class_labels, class_frequency, model_name)