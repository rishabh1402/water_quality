from flask import Flask, render_template, redirect, url_for, Response, request, session
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


app = Flask(__name__,template_folder='templates')

def graph(city):
    dp = pd.read_csv('D:/Hackathons/Delta/water/water.csv',encoding= 'unicode_escape')
    dp2 = dp[dp['District Name']==city].reset_index()
    sns.countplot(dp2['Quality Parameter'])
    plt.title('Water Quality Index', size=20)
    plt.savefig("D:/Hackathons/Delta/water/static/output.jpg")

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

# ROUTES
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/check", methods=["GET", "POST"])
def check():
    state = request.form['state']
    state = state.upper()
    city = request.form['city']
    city = city.upper()
    features = []
    features.append(state)
    features.append(city)
    output = predict(features)
    if output == 0:
        chem = "Arsenic"
    elif output == 1:
        chem = "Fluoride"
    elif output == 2:
        chem = "Iron"
    elif output == 3:
        chem = "Nitrate"
    else:
        chem = "Salt"
    graph(city)
    return render_template("success.html", p_text='The Dominant Chemical in the Water of your Area is  {}'.format(chem))    


@app.route("/about")
def about():
    return render_template("about.html")

"""
@app.route("/output.jpg")
def plot_png(city):
    graph(city)
    filename = 'output.jpg'
    return render_template("success.html",img = filename)
"""

if __name__ == "__main__":
    app.run(debug=True)