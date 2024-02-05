import pandas as pd
import numpy as np
from io import StringIO


#IO tools (text, CSV, HDF5, …)
data = "a,b,c,d\n1,2,3,4\n5,6,7,8\n9,10,11"

print(data)
df = pd.read_csv(StringIO(data), dtype=object)
df = pd.read_csv(StringIO(data), dtype={"b": object, "c": np.float64, "d": "Int64"})


#JSON
data = (
 '{"a":{"0":1,"1":3},"b":{"0":2.5,"1":4.5},"c":{"0":true,"1":false},"d":{"0":"a","1":"b"},'
 '"e":{"0":null,"1":6.0},"f":{"0":null,"1":7.5},"g":{"0":null,"1":true},"h":{"0":null,"1":"a"},'
 '"i":{"0":"12-31-2019","1":"12-31-2019"},"j":{"0":null,"1":null}}'
)


df = pd.read_json(StringIO(data))

jsonl = """
    {"a": 1, "b": 2}
    {"a": 3, "b": 4}
"""
df = pd.read_json(StringIO(jsonl), lines=True)


#HTML
#needs pip install lxml
url = "https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list"
pd.read_html(url)


#XML
xml = """<?xml version="1.0" encoding="UTF-8"?>
<bookstore>
  <book category="cooking">
    <title lang="en">Everyday Italian</title>
    <author>Giada De Laurentiis</author>
    <year>2005</year>
    <price>30.00</price>
  </book>
  <book category="children">
    <title lang="en">Harry Potter</title>
    <author>J K. Rowling</author>
    <year>2005</year>
    <price>29.99</price>
  </book>
  <book category="web">
    <title lang="en">Learning XML</title>
    <author>Erik T. Ray</author>
    <year>2003</year>
    <price>39.95</price>
  </book>
</bookstore>"""


df = pd.read_xml(StringIO(xml))


'''
#Excel files
pd.read_excel("path_to_file.xls", sheet_name="Sheet1")
with pd.ExcelFile("path_to_file.xls") as xls:
    df1 = pd.read_excel(xls, "Sheet1")
    df2 = pd.read_excel(xls, "Sheet2")
'''




#Pickling
df = pd.DataFrame(
    {"A": [1, 2, 3], "B": [4, 5, 6], "C": ["p", "q", "r"]}, index=["x", "y", "z"]
)
df.to_pickle("./data/saved.pkl")
pd.read_pickle("./data/saved.pkl")

df = pd.DataFrame(
    {
        "A": np.random.randn(1000),
        "B": "foo",
        "C": pd.date_range("20130101", periods=1000, freq="s"),
    }
)
df.to_pickle("./data/data.pkl.compress", compression="gzip")
rt = pd.read_pickle("./data/data.pkl.compress", compression="gzip")








