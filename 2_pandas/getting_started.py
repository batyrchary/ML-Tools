import pandas as pd


#The two primary data structures of pandas
#Series (1-dimensional) and DataFrame (2-dimensional)
#pandas is built on top of NumPy 


df = pd.DataFrame(
    {
        "Name": [
            "Braund, Mr. Owen Harris",
            "Allen, Mr. William Henry",
            "Bonnell, Miss. Elizabeth",
        ],
        "Age": [22, 35, 58],
        "Sex": ["male", "male", "female"],
    }
)
'''
print(df)
print(df["Age"])

age_column=df["Age"] 			#When selecting a single column of a pandas DataFrame, the result is a pandas Series

#You can create a Series from scratch as well:
ages = pd.Series([22, 35, 58], name="Age")

print(df["Age"].max())
print(ages.max())


#The describe() method provides a quick overview of the 
#numerical data in a DataFrame. As the Name and Sex columns are textual data
print(df.describe())
print("----------------")
'''


#--------Reading From File
titanic = pd.read_csv("./titanic.csv")
'''
print(titanic)
print("----------------")
small_titanic=titanic.head(8)
print(small_titanic) 
print(titanic.dtypes)
print(titanic.info())
print(titanic.describe())
'''



#--------Subset of DataFrame
print(titanic["Age"].shape)
age_sex = titanic[["Age", "Sex"]]  #filter specifica columns
print(age_sex.head(5))
print("----------------")

above_35 = titanic[titanic["Age"] > 35] #filter row
print(above_35.head(5))

print("----------------")
mask=titanic["Age"] > 35
print(mask.head(5))


#Titanic passengers from cabin class 2 and 3
class_23 = titanic[titanic["Pclass"].isin([2, 3])]
#class_23 = titanic[(titanic["Pclass"] == 2) | (titanic["Pclass"] == 3)] #similar as above 

#passenger data for which the age is known
age_no_na = titanic[titanic["Age"].notna()]


#select specific rows and columns from a DataFrame
#names of the passengers older than 35 years.
adult_names = titanic.loc[titanic["Age"] > 35, "Name"]


#rows 10 till 25 and columns 3 to 5
subset=titanic.iloc[9:25, 2:5]


#--------Plotting Data Frame
import matplotlib.pyplot as plt
#titanic.plot()
#plt.show()

#titanic["Age"].plot()
#plt.show()

#visually compare the values measured in pclass versus fare columns.
#titanic.plot.scatter(x="Pclass", y="Fare", alpha=0.5)
#plt.show()


titanic.plot.box()
plt.show()

