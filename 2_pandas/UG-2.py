import pandas as pd
import numpy as np

#Essential basic functionality

df = pd.DataFrame(
    {
        "one": pd.Series(np.random.randn(3), index=["a", "b", "c"]),
        "two": pd.Series(np.random.randn(4), index=["a", "b", "c", "d"]),
        "three": pd.Series(np.random.randn(3), index=["b", "c", "d"]),
    }
)

print(df)
row = df.iloc[1]
column = df["two"]

df_sub=df.sub(row, axis="columns")

print(row)
print(column)
print(df_sub)

print("-------------")

s = pd.Series(np.arange(10))
div, rem = divmod(s, 3)
print(div)
print(rem)

idx = pd.Index(np.arange(10))
print(idx)

div, rem = divmod(idx, 3)
print(div)
print(rem)
print("-------------")

df2=df*2

print(df)
print(df.gt(df2))
print(df2.ne(df))
print("-------------")

print((df > 0).all())
print((df > 0).any())

print(df.empty)
print("-------------")

#Comparing if objects are equivalent
print(df + df == df * 2)
print((df + df == df * 2).all())
print((df + df).equals(df * 2))

print(pd.Series(["foo", "bar", "baz"]) == "foo")
print(pd.Index(["foo", "bar", "baz"]) == "foo")
print(pd.Series(["foo", "bar", "baz"]) == pd.Index(["foo", "bar", "qux"]))
print(pd.Series(["foo", "bar", "baz"]) == np.array(["foo", "bar", "qux"]))

print("-------------")




#Combining overlapping data sets

df1 = pd.DataFrame(
    {"A": [1.0, np.nan, 3.0, 5.0, np.nan], "B": [np.nan, 2.0, 3.0, np.nan, 6.0]}
)

df2 = pd.DataFrame(
    {
        "A": [5.0, 2.0, 4.0, np.nan, 3.0, 7.0],
        "B": [np.nan, np.nan, 3.0, 4.0, 6.0, 8.0],
    }
)

df12=df1.combine_first(df2)
print(df1)
print("-------------")
print(df2)
print("-------------")
print(df12)
print("-------------")



#Descriptive statistics

print(df.sum(0, skipna=False))
print(df.sum(axis=1, skipna=True))

ts_stand = (df - df.mean()) / df.std()

print(ts_stand)
print("-------------")

frame = pd.DataFrame({"a": ["Yes", "Yes", "No", "No"], "b": range(4)})
frame.describe()
print("-------------")
frame.describe(include=["object"])
print("-------------")
frame.describe(include=["number"])
print("-------------")
frame.describe(include="all")
print("-------------")



#Index of min/max values
s1 = pd.Series(np.random.randn(5))
print(s1)
print(s1.idxmin(), s1.idxmax())
print("-------------")

df1 = pd.DataFrame(np.random.randn(5, 3), columns=["A", "B", "C"])
print(df1)
print(df1.idxmin(axis=0))
print(df1.idxmin(axis=1))
print("-------------")

#Value counts (histogramming) / mode
data = np.random.randint(0, 7, size=50)
s = pd.Series(data)
print(s.value_counts())


s5 = pd.Series([1, 1, 3, 3, 3, 5, 5, 7, 7, 7])
print(s5.mode())

df5 = pd.DataFrame(
    {
        "A": np.random.randint(0, 7, size=50),
        "B": np.random.randint(-10, 15, size=50),
    }
)
print(df5.mode())
print("-------------")


#Discretization and quantiling
arr = np.random.randn(20)
factor = pd.cut(arr, 4)
print(arr)
print(factor)
print("-------------")


#Row or column-wise function application
print(df.apply(lambda x: np.mean(x)))
print(df.apply(lambda x: np.mean(x), axis=1))
print(df.apply(lambda x: x.max() - x.min()))
print(df.apply(np.cumsum))
print(df.apply(np.exp))
print(df.apply("mean"))
print(df.apply("mean", axis=1))
print("-------------")

#Aggregation API
tsdf = pd.DataFrame(
    np.random.randn(10, 3),
    columns=["A", "B", "C"],
    index=pd.date_range("1/1/2000", periods=10),
)
print(tsdf)
print("-------------")
print(tsdf.agg(lambda x: np.sum(x)))
print(tsdf.agg("sum"))
print(tsdf["A"].agg("sum"))
print(tsdf.agg(["sum", "mean"]))
print(tsdf["A"].agg(["sum", "mean"]))
print("-------------")

print(tsdf.transform(np.abs))
print(tsdf.transform("abs"))
print(tsdf["A"].transform(np.abs))
print("-------------")


#Reindexing and altering labels
s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
s.reindex(["e", "b", "f", "d"])
df.reindex(index=["c", "f", "b"], columns=["three", "two", "one"])
df.reindex(["c", "f", "b"], axis="index")
df.reindex(["three", "two", "one"], axis="columns")

#Aligning objects with each other with align
s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
s1 = s[:4]
s2 = s[1:]

s1.align(s2)
s3=s1.align(s2, join="inner")
s4=s1.align(s2, join="left")

print(s3)
print("-------------")
print(s4)
print("-------------")


#Filling while reindexing

rng = pd.date_range("1/3/2000", periods=8)
ts = pd.Series(np.random.randn(8), index=rng)
ts2 = ts.iloc[[0, 3, 6]]
ts2.reindex(ts.index)
ts2.reindex(ts.index, method="ffill")
ts2.reindex(ts.index, method="bfill")
ts2.reindex(ts.index, method="nearest")
ts2.reindex(ts.index, method="ffill", limit=1)
print("-------------")


#Dropping labels from an axis
print(df.drop(["a", "d"], axis=0))
print(df.drop(["one"], axis=1))
print("-------------")


#Iteration
df = pd.DataFrame(
    {"col1": np.random.randn(3), "col2": np.random.randn(3)}, index=["a", "b", "c"]
)

for col in df:
    print(col)
print("-------------")

df = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
for index, row in df.iterrows():
    row["a"] = 10

#Sorting
df = pd.DataFrame(
    {
        "one": pd.Series(np.random.randn(3), index=["a", "b", "c"]),
        "two": pd.Series(np.random.randn(4), index=["a", "b", "c", "d"]),
        "three": pd.Series(np.random.randn(3), index=["b", "c", "d"]),
    }
)

unsorted_df = df.reindex(
    index=["a", "d", "c", "b"], columns=["three", "two", "one"]
)

unsorted_df.sort_index()
unsorted_df.sort_index(ascending=False)
unsorted_df.sort_index(axis=1)
unsorted_df["three"].sort_index()

s1 = pd.DataFrame({"a": ["B", "a", "C"], "b": [1, 2, 3], "c": [2, 3, 4]}).set_index(
    list("ab")
)

s1.sort_index(level="a")
s1.sort_index(level="a", key=lambda idx: idx.str.lower())

df1 = pd.DataFrame(
    {"one": [2, 1, 1, 1], "two": [1, 3, 2, 4], "three": [5, 4, 3, 2]}
)

df1.sort_values(by="two")

df1[["one", "two", "three"]].sort_values(by=["one", "two"])

s1 = pd.Series(["B", "a", "C"])
s1.sort_values()


df = pd.DataFrame({"a": ["B", "a", "C"], "b": [1, 2, 3]})
df.sort_values(by="a")


#smallest / largest values

s = pd.Series(np.random.permutation(10))
s.sort_values()
s.nsmallest(3)
s.nlargest(3)

df = pd.DataFrame(
    {
        "a": [-2, -1, 1, 10, 8, 11, -1],
        "b": list("abdceff"),
        "c": [1.0, 2.0, 4.0, 3.2, np.nan, 3.0, 4.0],
    }
)

df.nlargest(3, "a")
df.nlargest(5, ["a", "c"])
df.nsmallest(3, "a")
df.nsmallest(5, ["a", "c"])








