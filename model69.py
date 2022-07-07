
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data = pd.read_excel(r"D:\DS&AI\1DS\Project69\Dataset\final_data.xlsx")

data.head()

data.drop(['Unnamed: 0'], axis = 1, inplace = True)

data.info()

data.rename(columns={'shortest distance Agent-Pathlab(m)' : 'Distance Agent-Pathlab', 
                   'shortest distance Patient-Pathlab(m)' : 'Distance Patient-Pathlab',  
                   'shortest distance Patient-Agent(m)' : 'Distance Patient-Agent',  
                   'Availabilty time (Patient)' : 'Patient Availabilty',  
                   'Test Booking Date' : 'Booking Date',  
                   'Test Booking Time HH:MM' : 'Booking Time',
                   'Way Of Storage Of Sample' : 'Specimen Storage',
                   ' Time For Sample Collection MM' : 'Specimen collection Time',
                   'Time Agent-Pathlab sec' : 'Agent-Pathlab sec',
                   'Agent Arrival Time (range) HH:MM' : 'Agent Arrival Time',
                   'Exact Arrival Time MM' : 'Exact Arrival Time'   
                  }, inplace=True)

data.duplicated().any()

data.isna().any()

sns.distplot(data['Exact Arrival Time'])

id_columns = data[['Patient ID', 'Agent ID', 'pincode']]
num_columns = data[['Age', 'Distance Agent-Pathlab', 'Distance Patient-Pathlab', 'Distance Patient-Agent', 
                        'Specimen collection Time' , 'Agent-Pathlab sec', 'Exact Arrival Time']]
cat_columns = data[['patient location', 'Diagnostic Centers', 'Time slot', 'Patient Availabilty', 'Gender', 
                          'Booking Date', 'Specimen Storage', 'Sample Collection Date', 'Agent Arrival Time']]

list(cat_columns['Diagnostic Centers'].unique())

cat_columns['Diagnostic Centers'].value_counts().plot(kind = 'bar')

def name_change(text):
    if text == 'Medquest Diagnostics Center' or text == 'Medquest Diagnostics':
        return 'Medquest Diagnostics Center'
    elif text == 'Pronto Diagnostics' or text == 'Pronto Diagnostics Center':
        return 'Pronto Diagnostics Center'
    elif text == 'Vijaya Diagonstic Center' or text == 'Vijaya Diagnostic Center':
        return 'Vijaya Diagnostic Center'
    elif text == 'Viva Diagnostic' or text == 'Vivaa Diagnostic Center':
        return 'Vivaa Diagnostic Center'
    else:
        return text

cat_columns['Diagnostic Centers'] = cat_columns['Diagnostic Centers'].apply(name_change)

cat_columns['Diagnostic Centers'].value_counts().plot(kind = 'bar')

cat_columns['Time slot'].value_counts().plot(kind = 'bar')

cat_columns['Specimen Storage'].value_counts().plot(kind = 'bar')

cat_columns['Patient Availabilty'].value_counts().plot(kind = 'bar')

cat_columns['Gender'].value_counts().plot(kind = 'bar')

new_data = pd.concat([id_columns,
                    cat_columns[['Diagnostic Centers', 'Time slot', 'Patient Availabilty', 'Gender',
                                         'Specimen Storage', 'Agent Arrival Time']],
                    num_columns[['Distance Patient-Agent', 'Specimen collection Time', 'Exact Arrival Time']]
                   ], axis = 1)

final = new_data[new_data['Distance Patient-Agent'] != 0]

final.info()

sns.distplot(np.log(final['Distance Patient-Agent']))

final.drop(['Patient ID', 'pincode'], axis = 1, inplace = True)

final['Distance Patient-Agent'] = np.log(final['Distance Patient-Agent'])

final = final[final['Patient Availabilty'] != '19:00 to 22:00']

"""MODEL BUILDING"""

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC

final['Patient Availabilty From'] = final['Patient Availabilty'].apply(lambda x:x.split(':')[0])

a = final['Patient Availabilty'].apply(lambda x:x.split('to')[1])
final['Patient Availabilty To'] = a.apply(lambda x:x.split(':')[0])

b = final['Agent Arrival Time'].apply(lambda x:x.split('to')[1])
final['Agent Arrive Before'] = b.apply(lambda x:x.split(':')[0])

final['Patient Availabilty From'] = final['Patient Availabilty From'].astype('int64')
final['Patient Availabilty To'] = final['Patient Availabilty To'].astype('int64')
final['Agent Arrive Before'] = final['Agent Arrive Before'].astype('int64')

final_new = final.drop(['Patient Availabilty', 'Agent Arrival Time', 'Diagnostic Centers'], axis = 1)

final_new.head()

le = LabelEncoder()
final_new['Time slot'] = le.fit_transform(final_new['Time slot'])
final_new['Gender'] = le.fit_transform(final_new['Gender'])
final_new['Specimen Storage'] = le.fit_transform(final_new['Specimen Storage'])

variables = final_new.drop(['Exact Arrival Time'], axis = 1)
target = final_new[['Exact Arrival Time']]

xtrain, xtest, ytrain, ytest = train_test_split(variables, target, test_size=0.3)

lr = LogisticRegression(multi_class='ovr')
lr.fit(xtrain, ytrain)
ypred = lr.predict(xtest)
print('Accuracy score: {:.4f}'.format(accuracy_score(ytest, ypred))) 
print('Classification Report: \n', classification_report(ytest, ypred))

lr1 = LogisticRegression(multi_class='ovr',
                           penalty = 'l2',
                           solver='newton-cg',
                           C = 16.0,
                           fit_intercept=True,
                           class_weight='balanced',
                           random_state=50
                          ) 
lr1.fit(xtrain, ytrain)
ypred = lr1.predict(xtest)
print('Accuracy score: {:.4f}'.format(accuracy_score(ytest, ypred))) 
print('Classification Report: \n', classification_report(ytest, ypred))


import pickle

pickle.dump(lr1, open('logistic_reg.pkl', 'wb'))
pickle.dump(final_new, open('dataset.pkl', 'wb'))

