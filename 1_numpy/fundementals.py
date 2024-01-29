import numpy as np


#Advanced indexing
#The definition of advanced indexing means that x[(1, 2, 3),] is fundamentally 
#different than x[(1, 2, 3)]. The latter is equivalent to x[1, 2, 3] which will 
#trigger basic selection while the former will trigger advanced indexing. 


'''
#---------Indexing, Slicing
x = np.arange(10)
print(x)
print(x[2])
print(x[-2])

x=x.reshape(2,5)
print(x)
print(x[1,3], x[1][3]) #same thing


#Syntax is i:j:k where i is the starting index, j is the stopping index, and k is the step 
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y=x[1:7:2]

print(x[1:7:2])          #=>array([1, 3, 5])
print(x[-3:3:-1])        #=>array([7, 6, 5, 4])

print("-------------")
x = np.array([[[1],[2],[3]], [[4],[5],[6]]])
print(x)
print("-------------")
print(x[...,0])         #same
print(x[:, :, 0])       #same


print("-------------")
x = np.array([[ 0,  1,  2],
              [ 3,  4,  5],
              [ 6,  7,  8],
              [ 9, 10, 11]])

y1=x[1:2, 1:3]
y2=x[1:2, [1, 2]]
print(y1)   #=>[[4 5]]
print(y2)   #=>[[4 5]]
print("-------------")

#---------Assigning values to indexed arrays
x = np.arange(10)
x[2:7] = 1
print(x)
x[2:7] = np.arange(5)
print(x)
'''

'''
#---------Broadcasting

a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 2.0, 2.0])
c=a*b
print(c)


a = np.array([0.0, 10.0, 20.0, 30.0])
b = np.array([1.0, 2.0, 3.0])
a[:, np.newaxis] + b
#array([[ 1.,   2.,   3.],
#       [11.,  12.,  13.],
#       [21.,  22.,  23.],
#       [31.,  32.,  33.]])

#0          0  0  0
#10         10 10 10
#20         20 20 20
#30         30 30 30

#1 2 3       1 2 3
#            1 2 3
#            1 2 3
#            1 2 3
'''

'''
#---------Copy and View
#Views are created when elements can be addressed with offsets and strides in the original array. 
#Hence, basic indexing always creates views, y gets changed when x is changed because it is a view.
x = np.arange(10)
y = x[1:3]  # creates a view
print(x)
print(y)
x[1:3] = [10, 11]
print(x)    #=>array([ 0, 10, 11,  3,  4,  5,  6,  7,  8,  9])
print(y)    #=>array([10, 11])

#Advanced indexing, on the other hand, always creates copies. 
x = np.arange(9).reshape(3, 3)
y = x[[1, 2]]
print(x)
print(y)
#Here, y is a copy

#The base attribute of the ndarray makes it easy to tell if an array is a view or a copy.

if(y.base is None == True):
    print("y is a view")

'''

'''
#---------Structured arrays
#Here x is a one-dimensional array of length two whose datatype is a structure 
#with three fields: 1. A string of length 10 or less named ‘name’, 2. a 32-bit 
#integer named ‘age’, and 3. a 32-bit float named ‘weight’.

x = np.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)], dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
print(x)
print(x[1])
print(x['age'])     #=>array([9, 3], dtype=int32)
print("-------------")

x = np.array([(1, 2, 3), (4, 5, 6)], dtype='i8, f4, f8')
x[1] = (7, 8, 9)

x = np.array([(1, 2), (3, 4)], dtype=[('foo', 'i8'), ('bar', 'f4')])
print(x['foo'])     #=>array([1, 3])
x['foo'] = 10
print(x)
#array([(10, 2.), (10, 4.)],
#      dtype=[('foo', '<i8'), ('bar', '<f4')])

y = x['bar']
y[:] = 11
print(x)        #will modify the original array

#Accessing Multiple Fields
a = np.zeros(3, dtype=[('a', 'i4'), ('b', 'i4'), ('c', 'f4')])
print(a[['a', 'c']])    #=>array([(0, 0.), (0, 0.), (0, 0.)]

x = np.array([(1, 2), (3, 4)], dtype=[('foo', 'i8'), ('bar', 'f4')])
s = x[0]
print(s)        #(1, 2.)
s['bar'] = 100
print(s)        #(1, 100.)
print(x)        #=>array([(1, 100.), (3, 4.)],
'''

'''
#---------Structure Comparison and Promotion
a = np.array([(1, 1), (2, 2)], dtype=[('a', 'i4'), ('b', 'i4')])
b = np.array([(1, 1), (2, 3)], dtype=[('a', 'i4'), ('b', 'i4')])
print(a == b)       #=>array([True, False])
'''

#---------Ufunc methods
x = np.arange(9).reshape(3,3)
print(x)
#array([[0, 1, 2],
#      [3, 4, 5],
#      [6, 7, 8]])
print(np.add.reduce(x, 1))                      #=>array([ 3, 12, 21])
print(np.add.reduce(x, (0, 1)))                 #=>36
print(np.multiply.reduce(x, dtype=float))       #=>array([ 0., 28., 80.])

y = np.zeros(3, dtype=int)
print(y)                                            #=>array([0, 0, 0])
print(np.multiply.reduce(x, dtype=float, out=y))    #=>array([ 0, 28, 80])
