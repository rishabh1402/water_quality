import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
import mpld3







def graph():
    df = pd.read_csv('D:/Hackathons/Delta/water/water.csv')
    city = 'JORHAT'
    df2 = df[df['District Name']==city].reset_index()
    fig = figure()
    ax = fig.gca()
    ax.countplot([1,2,3,4])
    mpld3.show(fig)
    #sns.countplot(df2['Quality Parameter'])
    #fig = plt.title('Water Quality Parameter in Kerala', size=20)




