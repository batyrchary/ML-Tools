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


#Comparing if objects are equivalent
print(df + df == df * 2)
print((df + df == df * 2).all())
print("-------------")