import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])
b = a[:2]
b += 1
print('a =', a, '; b =', b)
#a = [2 3 3 4 5 6] ; b = [2 3]




A = np.ones((2, 2))
B = np.eye(2, 2)
C = np.zeros((2, 2))
D = np.diag((-3, -4))
R=np.block([[A, B], [C, D]])

print (R)


a=np.loadtxt('simple.csv', delimiter = ',', skiprows = 1) 
print(a)


#start:stop:step 
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(x[1:7:2])
#array([1, 3, 5])


#https://numpy.org/doc/stable/user/basics.indexing.html
#Dimensional indexing tools