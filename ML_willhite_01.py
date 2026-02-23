import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#initiation
titanic_data = pd.read_csv("C:/Users/mowma/Downloads/train.csv")
print(titanic_data.head(10))
print(titanic_data.info())
titanic_dummy = titanic_data.copy()
titanic_dummy = titanic_dummy.drop(columns=['Ticket', 'Cabin','Name'])
# some imputation

def format_sex(value):
    if value == 'male':
        return 0
    elif value=='female':
        return 1
    else:
        return value

def format_embarked(value):
    if value == "C":
        return 1
    elif value == "Q":
        return 2
    elif value == "S":
        return 3
    else:
        return value


titanic_dummy['Embarked'] = titanic_dummy['Embarked'].map(format_embarked)
titanic_dummy['Sex'] = titanic_dummy['Sex'].map(format_sex)
print(titanic_dummy.info())



#Start of EDA

plt.figure(figsize=(12,8))
sns.heatmap(titanic_dummy.corr(), cmap='coolwarm', annot=True, linewidth=.5)
plt.title("Titanic passenger attributes heatmap")
#plt.show()

#hist of attributes
titanic_dummy.hist(
    bins='auto',
    figsize=(15,20),
    layout=(5,10),
    edgecolor='black',
    grid=False,
    alpha=1
)

plt.title("Attribute histograms in titanic", fontsize=20)
plt.tight_layout(rect=[0,.03,1,.95])
#plt.show()

#Survival by age group
plt.figure(figsize=(10,6))
sns.histplot(data=titanic_dummy, x='Age', hue='Survived', multiple='stack', bins=16, palette='Set2')
plt.title('age disribution with survival stats')
plt.xlabel("age (5 years)")
plt.ylabel('count')
#plt.show()

plt.figure(figsize=(10,6))
sns.histplot(data=titanic_dummy, x='Pclass',hue='Survived', multiple='stack', bins=3, palette='Set2')
plt.title('Survival by class')
plt.xlabel('pClass')
plt.ylabel('count')
#plt.show()

plt.figure(figsize=(10,6))
sns.histplot(data=titanic_dummy, x='Sex', hue='Survived',multiple='stack', bins=2,palette="Set2")
plt.title('Survival by Sex (0=m, 1=F)')
plt.xlabel('Sex')
plt.ylabel('Count')
#plt.show()


# I am going to introduce a sort of elementary log odds 

base_survival = np.mean(titanic_dummy['Survived'])
base_rate = np.log(base_survival / (1- base_survival))

sex_survival = titanic_dummy.groupby('Sex')['Survived'].mean()
sex_weight = np.log(sex_survival/(1-sex_survival))


class_survival = titanic_dummy.groupby('Pclass')['Survived'].mean()
class_weight = np.log(class_survival / (1-class_survival))

# age isn't a categorical variable so You can process it the same way, here I bin age in order to categorize it as an age group, therefore allowing 
# us to apply our same logic that we used before
age_bins = [0,10,20,30,40,50,60,70,80]
age_labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80"]
titanic_dummy["AgeGroup"] = pd.cut(titanic_dummy["Age"], bins=age_bins, labels=age_labels, right=True)
age_survival = titanic_dummy.groupby('AgeGroup')['Survived'].mean()
age_weights = np.log(age_survival / (1-age_survival))
print("Survival by Age Group:\n", age_survival)


def predict_survival(sex, pclass, age):
    logit = base_survival
    logit += sex_weight[sex] - base_survival
    logit += class_weight[pclass] - base_survival

    if age <=10:
        age_group = "0-10"
    elif age <=20:
        age_group = "10-20"
    elif age <= 30:
        age_group = "20-30"
    elif age <= 40:
        age_group  = "30-40"
    elif age <= 50:
        age_group = "40-50"
    elif age <= 60:
        age_group = "50-60"
    elif age <= 70: 
        age_group = "60-70"
    elif age <= 80:
        age_group = "70-80"
    
    logit += age_weights[age_group] - base_survival
    probability_survival = 1/(1+np.exp(-logit))
    return probability_survival

#case one/comparsison
print("Models prediction for percent survival rate of a Man in third class age between 20 and 30", predict_survival(0,3,25))
actual_man_3 = titanic_dummy[
    (titanic_dummy['Sex'] ==0) &
    (titanic_dummy["Pclass"] ==1) &
    (titanic_dummy["Age"].between(20,30))
]
actual_man_survival = actual_man_3["Survived"].mean()
print("Actual survival rate" , actual_man_survival)

#case two/comparison
print("Model prediction for survival rate for a woman aged 20-30 in first class", predict_survival(1,1,15))
actual_woman = titanic_dummy[
    (titanic_dummy["Sex"]==0) &
    (titanic_dummy['Pclass'] ==1) &
    titanic_dummy['Age'].between(10,20)
]
actual_woman_survival = actual_woman['Survived'].mean()
print("Actual survival rate of the group", actual_woman_survival)