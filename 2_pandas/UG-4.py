import pandas as pd
import numpy as np
from io import StringIO


#Indexing and selecting data


dates = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.randn(8, 4),
               index=dates, columns=['A', 'B', 'C', 'D'])

print(df)
print("-------------")

s = df['A']
print(s[dates[5]])
print("-------------")

df[['B', 'A']] = df[['A', 'B']]
print(df)
print("-------------")

#df.loc[:, ['B', 'A']] = df[['A', 'B']] ##will not modify anything 
df.loc[:, ['B', 'A']] = df[['A', 'B']].to_numpy() #correct way of doing
print(df[['A', 'B']])
print("-------------")


#Attribute access
sa = pd.Series([1, 2, 3], index=list('abc'))
dfa = df.copy()

print(sa.b)
print(dfa.A)

sa.a = 5
print(sa)

dfa.A = list(range(len(dfa.index)))  # ok if A already exists
print(dfa)
print("-------------")


x = pd.DataFrame({'x': [1, 2, 3], 'y': [3, 4, 5]})
x.iloc[1] = {'x': 9, 'y': 99}
print(x)
print("-------------")


#df_new = pd.DataFrame({'one': [1., 2., 3.]})
#df_new.two = [4, 5, 6] ##will raise warning and will not to anything
#print(df_new)


#Slicing ranges
print(s)
print("-------------")
print(s[:5])
print("-------------")
print(s[::2])
print("-------------")
print(s[::-1])
print("-------------")
s2 = s.copy()
s2[:5] = 0
print(s2)
print("-------------")

print(df)
print("-------------")
print(df[:3])
print("-------------")
print(df[::-1])
print("-------------")



s1 = pd.Series(np.random.randn(6), index=list('abcdef'))
print(s1.loc['c':])
print(s1.loc['b'])
print("-------------")
s1.loc['c':] = 0
print(s1)
print("-------------")



df1 = pd.DataFrame(np.random.randn(6, 4),
                   index=list('abcdef'),
                   columns=list('ABCD'))

print(df1.loc[['a', 'b', 'd'], :])
print(df1.loc['d':, 'A':'C'])
print(df1.loc['a'])
print(df1.loc['a'] > 0)
print("-------------")

#Slicing with labels

s = pd.Series(list('abcde'), index=[0, 3, 2, 5, 4])
print(s.loc[3:5])

print(s.sort_index())
print(s.sort_index().loc[1:6])

s = pd.Series(list('abcdef'), index=[0, 3, 2, 5, 4, 2])
print(s.loc[3:5])
print("-------------")

#Selection by position
s1 = pd.Series(np.random.randn(5), index=list(range(0, 10, 2)))
print(s1)
print(s1.iloc[:3])
print(s1.iloc[3])
s1.iloc[:3] = 0
print(s1)
print("-------------")

df1 = pd.DataFrame(np.random.randn(6, 4),
                   index=list(range(0, 12, 2)),
                   columns=list(range(0, 8, 2)))

print(df1.iloc[:3])
print(df1.iloc[1:5, 2:4])
print(df1.iloc[[1, 3, 5], [1, 3]])
print(df1.iloc[1:3, :])
print(df1.iloc[:, 1:3])
print("-------------")


dfl = pd.DataFrame(np.random.randn(5, 2), columns=list('AB'))
dfl.iloc[:, 2:3]
dfl.iloc[:, 1:3]
dfl.iloc[4:6]
print("-------------")

#Selection by callable
df1 = pd.DataFrame(np.random.randn(6, 4),
                   index=list('abcdef'),
                   columns=list('ABCD'))


df1.loc[lambda df: df['A'] > 0, :]
df1.loc[:, lambda df: ['A', 'B']]
df1.iloc[:, lambda df: [0, 1]]
df1[lambda df: df.columns[0]]
df1['A'].loc[lambda s: s > 0]


#Selecting random samples
s = pd.Series([0, 1, 2, 3, 4, 5])
s.sample()
s.sample(n=3)
s.sample(frac=0.5)

# Without replacement (default):
s.sample(n=6, replace=False)
## With replacement:
s.sample(n=6, replace=True)
print("-------------")

#Setting with enlargement
dfi = pd.DataFrame(np.arange(6).reshape(3, 2),
                   columns=['A', 'B'])
print(dfi)
dfi.loc[:, 'C'] = dfi.loc[:, 'A']
print(dfi)
dfi.loc[3] = 5
print(dfi)

#Boolean indexing
s = pd.Series(range(-3, 4))
print(s[s > 0])
print(s[(s < -1) | (s > 0.5)])
print(s[~(s < 0)])

print(df[df['A'] > 0])

#Indexing with isin

s = pd.Series(np.arange(5), index=np.arange(5)[::-1], dtype='int64')
print(s.isin([2, 4, 6]))
print(s[s.isin([2, 4, 6])])


s[s > 0]
s.where(s > 0) #same as above


dates = pd.date_range('1/1/2000', periods=8)

df = pd.DataFrame(np.random.randn(8, 4),
                  index=dates, columns=['A', 'B', 'C', 'D'])

df[df < 0]
df.where(df < 0, -df)


#Mask
s.mask(s >= 0)
df.mask(df >= 0)
print("-------------")

#query

n = 10
df = pd.DataFrame(np.random.rand(n, 3), columns=list('abc'))
print(df)
print("-------------")
df[(df['a'] < df['b']) & (df['b'] < df['c'])]
df.query('(a < b) & (b < c)')

df = pd.DataFrame(np.random.randint(n, size=(n, 2)), columns=list('bc'))
df.query('index < b < c')



df = pd.DataFrame({'a': list('aabbccddeeff'), 'b': list('aaaabbbbcccc'),
                   'c': np.random.randint(5, size=12),
                   'd': np.random.randint(9, size=12)})

print("-------------")
df.query('a in b')
df[df['a'].isin(df['b'])]
df.query('a not in b')


#Duplicate data
df2 = pd.DataFrame({'a': ['one', 'one', 'two', 'two', 'two', 'three', 'four'],
                    'b': ['x', 'y', 'x', 'y', 'x', 'x', 'x'],
                    'c': np.random.randn(7)})
df2.duplicated('a')
df2.duplicated('a', keep='last')
df2.duplicated('a', keep=False)
df2.drop_duplicates('a')
df2.drop_duplicates('a', keep='last')
df2.drop_duplicates('a', keep=False)
df2.duplicated(['a', 'b'])
df2.drop_duplicates(['a', 'b'])
print("-------------")

#Missing values
idx1 = pd.Index([1, np.nan, 3, 4])
idx1.fillna(2)






