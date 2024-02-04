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


#titanic.plot.box()
#plt.show()



#--------Adding new column

titanic["age_multiplied"]=titanic["Age"]*2
print(titanic[["Age", "age_multiplied"]])
titanic["age_divided"]=titanic["Age"]/titanic["age_multiplied"]
print(titanic[["Age", "age_multiplied", "age_divided"]])

#renaming
titanic = titanic.rename(
    columns={
        "age_multiplied": "ageM",
        "age_divided": "ageD",
    }
)

print(titanic[["Age", "ageM", "ageD"]])
#titanic = titanic.rename(columns=str.lower)

print("----------------")

#--------Summary Statistics
print(titanic["Age"].mean())
print(titanic[["Age", "Fare"]].median())
print(titanic[["Age", "Fare"]].describe())


print(titanic.agg(
    {
        "Age": ["min", "max", "median", "skew"],
        "Fare": ["min", "max", "median", "mean"],
    }
))

#average age for male versus female Titanic passengers
print(titanic[["Sex", "Age"]].groupby("Sex").mean())

#mean ticket fare price for each of the sex and cabin class combinations
print(titanic.groupby(["Sex", "Pclass"])["Fare"].mean())

#number of passengers in each of the cabin classes?
print(titanic["Pclass"].value_counts())


#--------Reshaping
print(titanic.sort_values(by="Age"))

#sort the Titanic data according to the cabin class and age in descending order.
print(titanic.sort_values(by=['Pclass', 'Age'], ascending=False))


#--------Long to wide table format
'''
air_quality = pd.read_csv("data/air_quality_long.csv", index_col="date.utc", parse_dates=True)
no2 = air_quality[air_quality["parameter"] == "no2"]
no2_subset = no2.sort_index().groupby(["location"]).head(2)
#values for the three stations as separate columns next to each other.
no2_subset.pivot(columns="location", values="value")
print(no2.head())
#mean concentrations for NO2 and PM25 in each of the stations in table form.
air_quality.pivot_table(values="value", index="location", columns="parameter", aggfunc="mean")

#add new index with reset index
no2_pivoted = no2.pivot(columns="location", values="value").reset_index()

#collect all air quality measurements in a single column (long format).
no_2 = no2_pivoted.melt(id_vars="date.utc")
'''


#--------Combine data from multiple tables

air_quality_no2 = pd.read_csv("./air_quality_no2_long.csv", parse_dates=True)
air_quality_no2 = air_quality_no2[["date.utc", "location","parameter", "value"]]
print(air_quality_no2.head())

air_quality_pm25 = pd.read_csv("./air_quality_pm25_long.csv", parse_dates=True)
air_quality_pm25 = air_quality_pm25[["date.utc", "location","parameter", "value"]]
print(air_quality_pm25.head())


#The concat() function performs concatenation operations of multiple tables along one of the axes (row-wise or column-wise).
air_quality = pd.concat([air_quality_pm25, air_quality_no2], axis=0)
print(air_quality.head())
#In this specific example, the parameter column provided by the data ensures 
#that each of the original tables can be identified. This is not always the case. 
#The concat function provides a convenient solution with the keys argument, adding an 
#additional (hierarchical) row index

print("----------------")
air_quality_ = pd.concat([air_quality_pm25, air_quality_no2], keys=["aaa--PM25", "bbb--NO2"])

print(air_quality_)
print("----------------")



#--------Handling time series data
air_quality = pd.read_csv("./air_quality_no2_long.csv", parse_dates=True)
air_quality = air_quality.rename(columns={"date.utc": "datetime"})
air_quality["datetime"] = pd.to_datetime(air_quality["datetime"])

print(air_quality["datetime"].min()) 
print(air_quality["datetime"].max())

print(air_quality["datetime"].max() - air_quality["datetime"].min())
air_quality["month"] = air_quality["datetime"].dt.month
print(air_quality)


#What is the average concentration for NO2 each day of the week for each of the measurement locations
print(air_quality.groupby([air_quality["datetime"].dt.weekday, "location"])["value"].mean())

#Plot the typical NO2 pattern during the day of our time series of all stations together. 
#In other words, what is the average value for each hour of the day?
#fig, axs = plt.subplots(figsize=(12, 4))
#air_quality.groupby(air_quality["datetime"].dt.hour)["value"].mean().plot(kind='bar', rot=0, ax=axs)
#plt.show()




#--------Manipulate textual data
titanic["Name"].str.lower()
print(titanic["Name"]) #not doing in place
print(titanic["Name"].str.lower())

titanic_splitted=titanic["Name"].str.split(",")
print(titanic_splitted.head())

titanic_name_parsed=titanic["Surname"] = titanic["Name"].str.split(",").str.get(0)
print(titanic_name_parsed.head())


mask=titanic["Name"].str.contains("Countess")
print(mask.head())
print(titanic[titanic["Name"].str.contains("Countess")])

print(titanic["Name"].str.len())

#In the “Sex” column, replace values of “male” by “M” and values of “female” by “F”.
titanic["Sex_short"] = titanic["Sex"].replace({"male": "M", "female": "F"})
#titanic["Sex_short"] = titanic["Sex"].str.replace("female", "F")
#titanic["Sex_short"] = titanic["Sex_short"].str.replace("male", "M")



