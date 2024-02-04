import pandas as pd
import numpy as np

#Intro to Data Structure



#The axis labels are collectively referred to as the index.

data=["d1","d2","d3","d4"]
index=[1,2,3,4]
s = pd.Series(data, index=index)
print(s)
s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
print(s)
s = pd.Series(np.random.randn(5), index=[1, 2, 3, 4, 5])
print(s)

d = {"b": 1, "a": 0, "c": 2}
s=pd.Series(d)
print(s)


s=pd.Series(5.0, index=["a", "b", "c", "d", "e"])
print(s)

print("-----------")
data=[10, 11, 12, 13, 14]
index=[1, 2, 3, 4, 5]
s = pd.Series(data, index=index)

print(s.iloc[0])
print(s.iloc[:3])
print(s[s > s.median()])
print(s.iloc[[4, 3, 1]])
print(np.exp(s))
print("-----------")

snp=s.to_numpy()
print(snp)
print("-----------")


#Series is dict-like
s=pd.Series(5.0, index=["a", "b", "c", "d", "e"])
print(s["a"])
s["e"] = 12.0
print(s)
print("e" in s)
print("-----------")

ss=s+s
print(s)
print("-----------")
s2=s * 2
print(s2)
print("-----------")




#From dict of Series or dicts
d = {
    "one": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
    "two": pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"]),
}
df = pd.DataFrame(d)

print(df)
df1=pd.DataFrame(d, index=["d", "b", "a"])
print("-----------")
print(df1)
print("-----------")
df2=pd.DataFrame(d, index=["d", "b", "a"], columns=["two", "three"])
print(df2)
print("-----------")
print(df.index)
print("-----------")
print(df.columns)


#From a Series
ser = pd.Series(range(3), index=list("abc"), name="ser")
df=pd.DataFrame(ser)
print(df)

df["foo"] = "bar"
print("-----------")
print(df)

df["flag"] = df["ser"] > 1
print("-----------")
print(df)

    
del df["foo"]           #Columns can be deleted or popped like with a dict
print("-----------")
print(df)

three = df.pop("flag")
print("-----------")
print(df)




#Data alignment and arithmetic
df1 = pd.DataFrame(np.random.randn(10, 4), columns=["A", "B", "C", "D"])
df2 = pd.DataFrame(np.random.randn(7, 3), columns=["A", "B", "C"])

df=df1 + df2
print(df)
df=df - df.iloc[0]
print("-----------")
print(df)
df=df * 5 + 2
print("-----------")
print(df)


df1 = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]}, dtype=bool)
df2 = pd.DataFrame({"a": [0, 1, 1], "b": [1, 1, 0]}, dtype=bool)
print(df1 & df2)
print(df1 | df2)


dfT=df[:5].T            # only show the first 5 rows
print("-----------")
print(dfT)

ser1 = pd.Series([1, 2, 3], index=["a", "b", "c"])
ser2 = pd.Series([1, 3, 5], index=["b", "a", "c"])
print(np.remainder(ser1, ser2))

