import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('D:/Hackathons/Delta/water/water.csv',encoding= 'unicode_escape')

#Encoding The IteM Type Column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encoded_state = le.fit_transform(df['State Name'])
df['encoded_state'] = encoded_state

encoded_city = le.fit_transform(df['District Name'])
df['encoded_city'] = encoded_city

encoded_quality = le.fit_transform(df['Quality Parameter'])
df['encoded_quality'] = encoded_quality

feature_columns=['encoded_state', 'encoded_city']
x = df[feature_columns].values
y = df['encoded_quality'].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 3)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)

pred = classifier.predict(x_test)

"""
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
accuracy = accuracy_score(y_test, pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
"""

df2 = df.groupby('State Name').agg('first').reset_index()
df2 = df2.drop(['District Name','encoded_city','encoded_quality','Block Name','Quality Parameter','Panchayat Name', 'Village Name', 'Habitation Name', 'Year'], axis = 1)

df3 = df.groupby('District Name').agg('first').reset_index()
df3.drop(['State Name','encoded_state','encoded_quality','Block Name','Quality Parameter','Panchayat Name', 'Village Name', 'Habitation Name', 'Year'], axis = 1)



def predict(inp):
    state, city = inp
    st  = df2[df2['State Name'] == state].index[0]
    ct  = df3[df3['District Name'] == city].index[0]
    x_new=[[st,ct]]
    pred_new = classifier.predict(x_new)  
    return pred_new

sam = ['ASSAM','JORHAT']
predict(sam)