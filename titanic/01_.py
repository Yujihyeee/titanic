import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('data/train.csv')
# print(train.head(5))
# print(train.isnull().sum())


def show_pie_chart(df, col_name):
    colname_survived = survived_crosstab(df, col_name)
    pie_chart(colname_survived)
    return colname_survived


def survived_crosstab(df, col_name):
    feature_survived = pd.crosstab(df[col_name], df['Survived'])
    feature_survived.columns = feature_survived.columns.map({0: 'Dead', 1: 'Alive'})
    return feature_survived


def pie_chart(feature_survived):
    frows, fcols = feature_survived.shape
    pcol = 3
    prow = (frows/pcol + frows%pcol)
    plot_height = prow * 2.5
    plt.figure(figsize=(8, plot_height))

    for row in range(0, frows):
        plt.subplot(int(prow), int(pcol), int(row+1))
        index_name = feature_survived.index[row]
        plt.pie(feature_survived.loc[index_name], labels=feature_survived.loc[index_name].index, autopct='%1.1f%%')
        plt.title(f'{index_name} survived')
    plt.show()


c = show_pie_chart(train, 'Sex')
print(c)

c = show_pie_chart(train, 'Embarked')
print(c)

train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.')
train['Title'].value_counts()
train['Title'] = train['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dona', 'Dr',
                                'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir'], 'Other')
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'].value_counts()
c = show_pie_chart(train, 'Title')
c

meanAge = train[['Title', 'Age']].groupby(['Title']).mean()
for index, row in meanAge.iterrows():
    nullIndex = train[(train.Title == index) & (train.Age.isnull())].index
    train.loc[nullIndex, 'Age'] = row[0]

train['AgeCategory'] = pd.qcut(train.Age, 8, labels=range(1, 9))
train.AgeCategory = train.AgeCategory.astype(int)

c = show_pie_chart(train, 'AgeCategory')
c

train.Cabin.fillna('N', inplace=True)
train['CabinCategory'] = train['Cabin'].str.slice(start=0, stop=1)
train['CabinCategory'] = train['CabinCategory'].map({'N': 0, 'C': 1, 'B': 2, 'D': 3,
                                                     'E': 4, 'A': 5, 'F': 6, 'G': 7,
                                                     'T': 8})

c = show_pie_chart(train, 'CabinCategory')
c

train.Fare.fillna(0)
train['FareCategory'] = pd.qcut(train.Fare, 8, labels=range(1,9))
train.FareCategory = train.FareCategory.astype(int)

c = show_pie_chart(train, 'FareCategory')
c