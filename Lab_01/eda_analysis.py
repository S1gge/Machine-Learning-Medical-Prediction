import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 

# Hur många är positiva för hjärt-kärlsjukdom och hur många ̈ar negativa?
def plot_pos_neg(df):
    plot = plt.pie(df['cardio'].value_counts(), autopct='%1.2f%%',  explode=[0.1,0], shadow=True)
    plt.title('   Proportion of people who tests positive or negative for      \n cardiovascualr disease.    ')
    plt.legend(loc='best', labels=['Negative', 'Positive'])
    return plot

# Hur stor andel har normala, ̈over normala och långt ̈over normala kolesterolvärden?
def plot_cholesterol (df):

    plot = plt.pie(df['cholesterol'].value_counts(),  autopct='%1.2f%%', explode=[0.04,0.1,0.1], shadow=True)
    plt.title('Proportion of people who have normal, above normal and well \n above normal cholesterol levels.')
    plt.legend(loc='best', labels=['Normal','Above normal', 'Well above normal'])
    return plot

# Hur ser ålders fördelningen ut?
def plot_age(df):
    fig, ax = plt.figure(figsize=(6,4), dpi=100), plt.axes()
    plot = sns.histplot(data=[a/365 for a in df['age']])
    ax.set(xlabel='Age, years')
    return plot

# Hur ser längd fördelningen ut?
def plot_height(df):
    fig, ax = plt.figure(figsize=(6,4), dpi=100), plt.axes()
    plot = sns.histplot(data=df['height'], bins=80)
    ax.set(xlabel='Age (years)')
    return plot

# Hur stor andel av kvinnor respektive män har hjärt-kärl sjukdom
def plot_men_women (df):
    df_pos = df[df['cardio']==1]
    df_pos_g = df_pos['gender'].value_counts()
    plot = plt.pie(df_pos_g, autopct='%1.2f%%', explode=[0.1,0], shadow=True)
    plt.title('Proportion of women and men who have cardiovascular disease.')
    plt.legend(loc='best', labels=['Male','Female'])
    return plot