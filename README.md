import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn

## ***Churn model***

df = pd.read_csv("/content/StudentsPerformance.csv")

df

df.info()

df.describe()

df[df['gender'] == 'male']['race/ethnicity'].value_counts()

df[df['gender'] == 'female']['race/ethnicity'].value_counts()

df.groupby(['gender', 'race/ethnicity', 'lunch', 'parental level of education']).agg({'reading score':['median', 'mean','max', 'min'], 'math score':['median', 'mean','max', 'min'], 'writing score':['median', 'mean','max', 'min']}, numeric_only=True)

plt.figure(figsize=(12, 3), dpi=100)
plt.title('Иллюстрация', color='red', fontsize=18)
plt.plot(df['math score'], color='black', linestyle='-', linewidth=0.8, label='')
plt.xlabel('Точки наблюдения', color='grey', fontsize=12)
plt.ylabel('', color='blue', fontsize=12, rotation=90)
plt.xticks(np.arange(0, 1000, 250),fontsize=9, color='green', rotation=90)
plt.yticks(np.arange(0, 2250, 500),fontsize=9, color='green', rotation=45)
plt.grid(color='red', alpha=0.3)
plt.legend(loc=0, fontsize=12, framealpha=0.9)
plt.savefig('Рисунок 1.png')
plt.show()

plt.figure(figsize=(12, 3), dpi=100)
plt.title('Иллюстрация', color='red', fontsize=18)
plt.plot(df['reading score'], color='black', linestyle='-', linewidth=0.8, label='')
plt.xlabel('Точки наблюдения', color='grey', fontsize=12)
plt.ylabel('', color='blue', fontsize=12, rotation=90)
plt.xticks(np.arange(0, 1000, 250),fontsize=9, color='green', rotation=90)
plt.yticks(np.arange(0, 2250, 500),fontsize=9, color='green', rotation=45)
plt.grid(color='red', alpha=0.3)
plt.legend(loc=0, fontsize=12, framealpha=0.9)
plt.savefig('Рисунок 1.png')
plt.show()

plt.figure(figsize=(12, 4))

plt.bar(df['race/ethnicity'].value_counts().index, df['City'].value_counts(), color='b')

plt.title('', fontsize=16, color='black')
plt.xlabel('', fontsize=12, color='grey')
plt.xticks(fontsize=12, color='red', rotation=0)
plt.yticks(fontsize=12, color='red', rotation=0)
plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.3)
plt.savefig('Рисунок 8. Bar plot for "race.jpeg', dpi=300)
plt.show()

df.corr(numeric_only=True)

plt.figure(figsize=(12, 3))
plt.title('race/ethnicity', color='red', fontsize=18)


plt.scatter(x=df['parental level of education'], y=df[''], color='black', )


plt.xlabel('', color='blue', fontsize=12)
plt.ylabel('', color='blue', fontsize=12, rotation=90)
plt.xticks(fontsize=9, color='green', rotation=90)
plt.yticks(fontsize=9, color='green', rotation=45)
plt.grid(color='red', alpha=0.3)
plt.legend(loc=0, fontsize=12, framealpha=0.9)
plt.savefig('Рисунок 1.png')
plt.show()

df.head()

df.info()

df.head()

df['gender'].value_counts()

df[['male', 'female']] = pd.get_dummies(df['gender'], dtype=int)

df.drop(columns=['gender'], inplace=True)

df['race/ethnicity'].value_counts()

df[['group C', 'group D', 'group B', 'group E', 'group A']] = pd.get_dummies(df['race/ethnicity'], dtype=int)

df.drop(columns=['race/ethnicity'], inplace=True)

df['parental level of education'].value_counts()

df[['some college', "associate's degree ", 'high school' , 'some high school', 'bachelor's degree', 'master's degree']] = pd.get_dummies(df['race/ethnicity'], dtype=int)

df.columns

