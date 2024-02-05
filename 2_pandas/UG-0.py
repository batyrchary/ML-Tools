import pandas as pd
import numpy as np

#10 minutes to pandas

#--------Object creation
#creating series
s = pd.Series([1, 3, 5, np.nan, 6, 8])


df = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp("20130102"),
        "C": pd.Series(1, index=list(range(4)), dtype="float32"),
        "D": np.array([3] * 4, dtype="int32"),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "foo",
    }
)
#print(df)


#--------Viewing data

print(df.head())
print(df.tail(-2))

ids = [1,2,3,4,5,6]
df = pd.DataFrame(np.random.randn(6, 4), index=ids, columns=list("ABCD"))

numpyV=df.to_numpy()
print(numpyV)

print(df.describe())
transposeDF=df.T 		#taking transpose
print(transposeDF)


print("-------------")
print(df)
print("-------------")
print(df.index)
print("-------------")
print(df.columns)
print("-------------")
print(df.sort_index(axis=1, ascending=False))
print(df.sort_values(by="B"))


#--------Selection
print("-------------")
print(df["A"])
print("-------------")
print(df[0:3])
print("-------------")
#Selecting a row matching a label:
print(df.loc[ids[0]])
print("-------------")
#Selecting all rows (:) with a select column labels:
print(df.loc[:, ["A", "B"]])

#Selecting a single row and column label returns a scalar:
df.loc[ids[0], "A"]

#--------Selection Via position
print(df.iloc[3])
print("-------------")
print(df.iloc[3:5, 0:2])
print("-------------")
print(df.iloc[[1, 2, 4], [0, 2]])	#Lists of integer position locations
print("-------------")
print(df.iloc[1:3, :])		#slicing rows explicitly
print("-------------")
print(df.iloc[:, 1:3])		#slicing columns explicitly
print("-------------")
print(df.iloc[1, 1])		#For getting a value explicitly:
print(df.iat[1, 1])			#equivalent to the prior method



#--------Boolean indexing
print(df[df["A"] > 0])
print(df[df > 0])			#Selecting values from a DataFrame where a boolean condition is met:

#Using isin() method for filtering:
df2 = df.copy()
df2["E"] = ["one", "one", "two", "three", "four", "three"]

print(df2)
print(df2[df2["E"].isin(["two", "four"])])

print("-------------")


#--------Setting

df.at[ids[0], "A"] = 0
df.iat[1, 1] = 10
df.loc[:, "D"] = np.array([5] * len(df))
print(df)

df2 = df.copy()

df2[df2 > 0] = -df2
print("-------------")
print(df2)

#--------Missing Data
df1 = df.reindex(index=ids[0:4], columns=list(df.columns) + ["E"])
df1.loc[ids[0] : ids[1], "E"] = 1
print(df1)
print("-------------")

df2=df1.dropna(how="any")
print(df2)
print("-------------")

df1=df1.fillna(value=5)
print(df2)
print("-------------")

mask=pd.isna(df1)




#--------Stats
#Operations in general exclude missing data.
print(df.mean()) 		#Calculate the mean value for each column
print(df.mean(axis=1))	#Calculate the mean value for each row




print(df.agg(lambda x: np.mean(x) * 5.6)) 
print(df.transform(lambda x: x * 101.2))

s = pd.Series(np.random.randint(0, 7, size=10))
s.value_counts()

s = pd.Series(["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"])
s.str.lower()


#--------Merge

df = pd.DataFrame(np.random.randn(10, 4))
pieces = [df[:3], df[3:7], df[7:]]
df_from_pieces=pd.concat(pieces)

print(df)
print("-------------")
print(df_from_pieces)


print("-------------")
left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})

print(left)
print(right)
merged=pd.merge(left, right, on="key")
print(merged)
print("-------------")




left = pd.DataFrame({"key": ["foo", "bar"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "bar"], "rval": [4, 5]})
print(left)
print(right)
merged=pd.merge(left, right, on="key")
print(merged)
print("-------------")


df = pd.DataFrame(
    {
        "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
        "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
        "C": np.random.randn(8),
        "D": np.random.randn(8),
    }
)

print(df)
print("-------------")
grouped_summed1=df.groupby("A")[["C", "D"]].sum()
print(grouped_summed1)
print("-------------")
grouped_summed2=df.groupby(["A", "B"]).sum()
print(grouped_summed2)
print("-------------")

#--------Stack

arrays = [
   ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
   ["one", "two", "one", "two", "one", "two", "one", "two"],
]

index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=["A", "B"])
df2 = df[:4]
stacked = df2.stack(future_stack=True)
print(stacked)
print("-------------")
unstacked=stacked.unstack()
print(unstacked)
print("-------------")
unstacked0=stacked.unstack(0)
print(unstacked0)
print("-------------")
unstacked1=stacked.unstack(1)
print(unstacked1)
print("-------------")


#--------Categoricals

df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": ["a", "b", "b", "a", "a", "e"]})
df["grade"] = df["raw_grade"].astype("category")
print(df)
new_categories = ["very good", "good", "very bad"]
df["grade"] = df["grade"].cat.rename_categories(new_categories)
print("-------------")
print(df)
df=df.groupby("grade", observed=False).size()
print("-------------")
print(df)
