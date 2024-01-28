import numpy as np




#---------Installing NumPy
#pip install numpy




#---------Array
#rank of the array is the number of dimensions
#shape of the array is a tuple of integers giving the size of the array along each dimension.

'''
a = np.array([1, 2, 3, 4, 5, 6])
print (a.shape)
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print (a.shape)
print(a[1])
print(a[1][1])
'''

'''
#---------Create Array
a = np.array([2, 1, 3])     #create array 1 2 3
print (a)
b=np.zeros(2)               #fill with zeros
print (b)
c=np.ones(2)                #fill with ones
print (c)
d=np.empty(2)               #empty array with two random elements elements
print (d)
e=np.arange(4)              #array with 0 1 2 3
print (e)
f=np.arange(2, 9, 2)        #first number, last number, and the step size. 2 4 6 8
print (f)
h=np.linspace(0, 10, num=5)   #spaced linearly 0, 2.5, 5, 7.5, 10
print (h)
c=np.ones(2, dtype=np.int64)  #can specify data type
print (c)
'''


'''
#---------Adding Removing Sorting

a1D = np.array([1, 2, 5, 3, 4])
a2D = np.array([[1, 2], [3, 4]])
a3D = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

#SORT
a1D=np.sort(a1D)
print (a1D)

#argsort, which is an indirect sort along a specified axis,
#lexsort, which is an indirect stable sort on multiple keys,
#searchsorted, which will find elements in a sorted array, and
#partition, which is a partial sort.


#CONCATENATE
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
c=np.concatenate((a, b))
print (c)

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6]])

c=np.concatenate((x, y))
print (c)
'''

'''
#---------Size, Shape, Dimension, Reshape
#ndarray.ndim will tell you the number of axes, or dimensions, of the array.
#ndarray.size will tell you the total number of elements of the array. 

#ndarray.shape will display a tuple of integers that indicate the number of 
#   elements stored along each dimension of the array. If, for example, you have a 2-D array 
#   with 2 rows and 3 columns, the shape of your array is (2, 3).

array_example = np.array([[[0, 1, 2, 3],
                           [4, 5, 6, 7]],
                          [[0, 1, 2, 3],
                           [4, 5, 6, 7]],
                          [[0 ,1 ,2, 3],
                           [4, 5, 6, 7]]])

print (array_example.ndim)
print (array_example.size)
print (array_example.shape)


# If you start with an array with 12 elements, you’ll need to make 
#sure that your new array also has a total of 12 elements.
a = np.arange(6)
b = a.reshape(3, 2) #array with 3 rows and 2 columns
print(b)
'''


'''
#---------Convert 1D array to 2D array
a = np.array([1, 2, 3, 4, 5, 6])
print(a.shape)
print(a)
row_vector = a[np.newaxis, :]   #np.newaxis to add a new axis:
col_vector = a[:, np.newaxis]   #np.newaxis to add a new axis:
print(row_vector.shape)
print(col_vector.shape)
print (row_vector)
print (col_vector)
'''


'''
#---------Indexing and Slicing
data = np.array([1, 2, 3])
a1=data[1]
a2=data[0:2]
a3=data[1:]
a4=data[-2:]


print (a1)
print (a2)
print (a3)
print (a4)


print(data<2)
print (data[data<2]) # all items less than 2

condition=((data<2) | (data>2))
print (condition)
print (data[condition]) # all items less or greater than 2 

'''

'''
#---------hstacking and vstacking,hsplit
a1 = np.array([[1, 1],
               [2, 2]])
a2 = np.array([[3, 3],
               [4, 4]])


c=np.vstack((a1, a2))
print(c)
#array([[1, 1],
#       [2, 2],
#       [3, 3],
#       [4, 4]])


c=np.hstack((a1, a2))
print(c)
#array([[1, 1, 3, 3],
#       [2, 2, 4, 4]])


x = np.arange(1, 25).reshape(2, 12)
print(x)
#array([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
#       [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]])


x=np.hsplit(x, 3)
print(x)
#[
# array([[ 1,  2,  3,  4],
#        [13, 14, 15, 16]]), 
# array([[ 5,  6,  7,  8],
#        [17, 18, 19, 20]]), 
# array([[ 9, 10, 11, 12],
#        [21, 22, 23, 24]])
#]
'''


'''
#---------Deep and Shallow copy

#A will also be modified if B is modified, B is shallow copy of A
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
b1 = a[0, :]
print(b1)
#array([1, 2, 3, 4])

b1[0] = 99
print(b1)
#array([99,  2,  3,  4])
print(a)
#array([[99,  2,  3,  4],
#       [ 5,  6,  7,  8],
#       [ 9, 10, 11, 12]])

b=a.copy() #is a deep copy
'''

'''
#---------Array operations
data = np.array([1, 2])
ones = np.ones(2, dtype=int)
c=data + ones
print(c)
#array([2, 3])
#1      1       2
#   +       = 
#2      1       2

c=data - ones
#print (c)
#array([0, 1])

c=data * data
#print (c)  
#array([1, 4])

c=data / data
#print (c)
#array([1., 1.])

#1      1       0
#   -       = 
#2      1       1



#1      1       1
#   *       = 
#2      2       4


#1      1       1
#   /       = 
#2      2       1


a = np.array([1, 2, 3, 4])
#print(a.sum())
#10
#print(a.max())
#print(a.min())

b = np.array([[1, 1], [2, 2]])
#print(b.sum(axis=0))
#array([3, 3])

#print(b.sum(axis=1))
#array([2, 4])


data = np.array([1.0, 2.0])
#print(data * 1.6)
#array([1.6, 3.2])

m1=data.max()
m2=data.min()
m3=data.sum()
data = np.array([[1, 2], [5, 4], [3, 6]])
m4=data.max(axis=0)         #maximum value within each column by specifying axis=0.
print (m1,m2,m3, m4)
'''

'''
#---------Creating matrices
data = np.array([[1, 2], [3, 4], [5, 6]])
print(data)
#array  (   [[1, 2],
#           [3, 4],
#           [5, 6]]
#       )

print(data[0, 1])
#2
print(data[1:3])
#array([[3, 4],
#       [5, 6]])
print(data[0:2, 0])
#array([1, 3])

#   0   0
#0  1   2
#1  3   4
#2  5   6

#print(data[0, 1])
#   0   0
#0  _   2
#1  _   _
#2  _   _

#print(data[1:3])
#   0   0
#0  _   _
#1  3   4
#2  5   6


#print(data[0:2, 0])
#   0   0
#0  1   _
#1  3   _
#2  _   _


print(data.max(axis=0))
#array([5, 6])
print(data.max(axis=1))
#array([2, 5, 6])


data = np.array([[1, 2], [3, 4]])
ones = np.array([[1, 1], [1, 1]])
print(data + ones)
#array([[2, 3],
#       [4, 5]])

#ones_row will be broadcasted
data = np.array([[1, 2], [3, 4], [5, 6]])
ones_row = np.array([[1, 1]])
print(data + ones_row)
#array([[2, 3],
#       [4, 5],
#       [6, 7]])


np.ones((3, 2))
np.zeros((3, 2))
rng = np.random.default_rng()
rng.random((3, 2)) 

'''



#---------Unique and Count

a = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])
unique_values = np.unique(a)
print(unique_values)
#[11 12 13 14 15 16 17 18 19 20]

unique_values, indices_list = np.unique(a, return_index=True)
print(indices_list)
#[ 0  2  3  4  5  6  7 12 13 14]

unique_values, occurrence_count = np.unique(a, return_counts=True)
print(occurrence_count)
#[3 2 2 2 1 1 1 1 1 1]

print("-----------")


a_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 3, 4]])
print(a_2d)
unique_values = np.unique(a_2d)
print(unique_values)
#[ 1  2  3  4  5  6  7  8  9 10 11 12]
#If the axis argument isn’t passed, your 2D array will be flattened.

unique_rows = np.unique(a_2d, axis=0)
print(unique_rows)
#[[ 1  2  3  4]
# [ 5  6  7  8]
# [ 9 10 11 12]]


unique_rows, indices, occurrence_count = np.unique(a_2d, axis=0, return_counts=True, return_index=True)
print(unique_rows)
#[[ 1  2  3  4]
#[ 5  6  7  8]
#[ 9 10 11 12]]
print(indices)
#[0 1 2]
print(occurrence_count)
#[2 1 1]


'''
#---------Transposing and Reshaping
arr = np.arange(6).reshape((2, 3))
print(arr)
#array([[0, 1, 2],
#       [3, 4, 5]])


arr=arr.transpose()
print(arr)
arr=arr.T
print(arr)
#array([[0, 3],
#       [1, 4],
#       [2, 5]])
'''



'''
#---------Reverse Array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
reversed_arr = np.flip(arr)
print('Reversed Array: ', reversed_arr)

arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
reversed_arr = np.flip(arr_2d)
print(reversed_arr)


# reverse only the columns or rows
reversed_arr_rows = np.flip(arr_2d, axis=0)
print(reversed_arr_rows)
reversed_arr_columns = np.flip(arr_2d, axis=1)
print(reversed_arr_columns)


#reverse the contents of only one column or row.
arr_2d[1] = np.flip(arr_2d[1])
print(arr_2d)


'''






'''











Burda kaldim #### Reshaping and flattening multidimensional arrays


print("-----------")


x = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
#x=x.flatten()
#print(x)

a1 = x.flatten()
a1[0] = 99
print(x)  # Original array
#[[ 1  2  3  4]
# [ 5  6  7  8]
# [ 9 10 11 12]]
print(a1)  # New array
#[99  2  3  4  5  6  7  8  9 10 11 12]

#when you use ravel, the changes you make to the new array will affect the parent array.
a2 = x.ravel()
a2[0] = 98
print(x)  # Original array
#[[98  2  3  4]
# [ 5  6  7  8]
# [ 9 10 11 12]]
print(a2)  # New array
#[98  2  3  4  5  6  7  8  9 10 11 12]



a = np.array([1, 2, 3, 4, 5, 6])
np.save('./filename', a)
b = np.load('./filename.npy')
print(b)

csv_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
np.savetxt('new_file.csv', csv_arr)
b=np.loadtxt('new_file.csv')
print(b)


print("------------")
a = np.array([1, 2, 3, 4, 5, 6])
b=a.reshape(3,2)

for row in b:
    print(row)


for element in b.flat:
    print(element)

'''

