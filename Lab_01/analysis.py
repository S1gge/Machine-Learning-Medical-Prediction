import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 


###################### PLottning ############################

# plt.pie cardio, cholesterol, smoke, height
def eda_pie(df):
    fig,axs = plt.subplots(2, 2, figsize = (12,14))

    axs[0,0].pie(df['cardio'].value_counts(), autopct='%1.2f%%',  explode=[0.1,0], shadow=True)
    axs[0,0].set_title('Proportion of people who tests positive or negative for \n cardiovascualr disease.')
    axs[0,0].legend(loc='best', labels=['Negative', 'Positive'])

    axs[0,1].pie(df['cholesterol'].value_counts(),  autopct='%1.2f%%', explode=[0.04,0.1,0.1], shadow=True)
    axs[0,1].set_title('Proportion of people who have normal, above normal and well \n above normal cholesterol levels.')
    axs[0,1].legend(loc='best', labels=['Normal','Above normal', 'Well above normal'])

    axs[1,0].pie(df['smoke'].value_counts(), autopct='%1.2f%%', explode=[0.2,0], shadow=True)
    axs[1,0].set_title('Proportion of people who smoke or not.')
    axs[1,0].legend(loc='best', labels=["Don't smoke","Smoke"])

    axs[1,1].pie(df[df['cardio']==1]['gender'].value_counts(), autopct='%1.2f%%', explode=[0.1,0], shadow=True)
    axs[1,1].set_title('Proportion of women and men who have cardiovascular disease.')
    axs[1,1].legend(loc='best', labels=['Male','Female'])

    return

# histplot age, weight, height
def eda_hist(df):
    fig,axs = plt.subplots(3,1, figsize = (10,10))

    sns.histplot(data=[a/365 for a in df['age']], bins=30, ax=axs[0])
    axs[0].set_title('Age distribution.')
    axs[0].set_xlabel('Age, years.')

    sns.histplot(data=df['weight'], bins=30, ax=axs[1])
    axs[1].set_title('Weight destripution.')
    axs[1].set_xlabel('Weight, kg')

    sns.histplot(data=df['height'], bins=30, ax=axs[2])
    axs[2].set_title('Height distribution.')
    axs[2].set_xlabel('Height, cm.')

    plt.tight_layout()

    return


# Visualiseringar andel sjukdomar
def vis_disease(df):
    fig, axes = plt.subplots(3, 2, dpi=100, figsize=(12, 14))
    axes = axes.flatten()

    df_h=df.copy()
    df_h['cholesterol']=df_h['cholesterol'].replace({1:'Normal', 2:'Above normal', 3:'Well above normal'})
    df_h['gluc']=df_h['gluc'].replace({1:'Normal', 2:'Above normal', 3:'Well above normal'})
    df_h['alco']=df_h['alco'].replace({1:'Alcohol Consumer', 0:'Non-Alcohol Consumer'})
    df_h['smoke']=df_h['smoke'].replace({1:'Smoker', 0:'Non-Smoker'})
    hues = ['bmi_cat', 'pressure_cat', 'cholesterol','gluc', 'alco', 'smoke']
    titles = ['BMI Categories', 'Pressure Categories', 'Cholesterol Levels', 
              'Glucose levels', 'Alcohol Consumption', 'Smoke Status']

    for i, (hue, title) in enumerate(zip(hues, titles)):
       plot = sns.countplot(df_h[df_h['cardio']==1], x='cardio', hue=hue, ax=axes[i])
       axes[i].set_ylabel('Count', fontsize = 10)
       axes[i].set_title(f"Distribution of {title} in \n Patients With Cardiovascular Disease. ", fontsize=13)
       axes[i].get_legend().set_title('')
    plt.tight_layout()
    plt.show()
    return plot

# Heatmap av korrelatione
def plot_heat(df):
    df = pd.get_dummies(df, drop_first=True)
    plt.figure(figsize=(10,8))
    plot = sns.heatmap(df.corr(), annot=True, annot_kws={'size':8}, fmt= '.2f', cmap='coolwarm', linewidths=0.5)
    return plot



################### Feature_engineering#######################
def iqr(df, feature, threshold):
    Q1, Q3 = np.quantile(df[feature], 0.25), np.quantile(df[feature], 0.75)
    IQR = Q3 - Q1
    return df[(df[feature] >= Q1 - threshold * IQR) & ( df[feature] <= Q3 + threshold * IQR)]


# bmi
def bmi(df):
    df['bmi'] = round(np.divide(df['weight'],np.power(df['height']*0.01,2)))
    return df['bmi']


# bmi category
def bmi_cat(df):
    df['bmi_cat'] = pd.cut(x=df['bmi'], bins=[18.5,25,30,35,40,99],
                         labels = ['Normal range', 'Over-weight', 'Obese (class 1)','Obese (class 2)', 'Obese (class 3)'])
    return df

# blood pressure category (medium.com)
def pressure_cat(df):
    
    conditions = [
        ((df['ap_hi'] <= 120) & (df['ap_lo'] <= 80)),
        ((df['ap_hi'] > 120) & (df['ap_hi'] <= 129)) & ((df['ap_lo'] >= 60) & (df['ap_lo'] <= 80)),
        ((df['ap_hi'] >= 130) & (df['ap_hi'] <= 139)) | ((df['ap_lo'] > 80) & (df['ap_lo'] <= 89)),
        ((df['ap_hi'] >= 140) & (df['ap_hi'] <= 179)) | ((df['ap_lo'] >= 90) & (df['ap_lo'] <= 119)),
        ((df['ap_hi'] >= 180) | (df['ap_lo'] >= 120))  
    ]
    choices = [
        'Healthy', 
        'Elevated', 
        'Stage 1 hypertension', 
        'Stage 2 hypertension', 
        'Hypertension crysis']
    
    df['pressure_cat'] = np.select(conditions, choices, default='Unknown')
       
    return df