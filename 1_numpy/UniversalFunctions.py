import numpy as np

#Functions that operate element by element on whole arrays.

#---------Reduce
#Reduces array’s dimension by one, by applying ufunc along one axis.
np.multiply.reduce([2,3,5])
X = np.arange(8).reshape((2,2,2))
a=np.add.reduce(X, 0)
b=np.add.reduce(X)
c=np.add.reduce(X, 1)
d=np.add.reduce(X, 2)
e=np.add.reduce([10], initial=5)  #The value with which to start the reduction.

print(X)
print("--------")
print(a)
print("--------")
print(b)
print("--------")
print(c)
print("--------")
print(d)
print("--------")
print(e)
print("++++++++")

#---------Accumulate
#Accumulate the result of applying the operator to all elements.
a=np.add.accumulate([2, 3, 5])
b=np.multiply.accumulate([2, 3, 5])
print(a)
print("--------")
print(b)
print("++++++++")

#---------ReduceAt
#Performs a (local) reduce with specified slices over a single axis.
a=np.add.reduceat(np.arange(8),[0,4, 1,5, 2,6, 3,7])[::2]     #=>array([ 6, 10, 14, 18])

#To take the running sum of four successive values:
b = np.linspace(0, 15, 16).reshape(4,4)
#array([[ 0.,   1.,   2.,   3.],
#       [ 4.,   5.,   6.,   7.],
#       [ 8.,   9.,  10.,  11.],
#       [12.,  13.,  14.,  15.]])
c=np.add.reduceat(b, [0, 3, 1, 2, 0])
print(a)
print("--------")
print(b)
print("--------")
print(c)
print("++++++++")


#---------Outer
#Apply the ufunc op to all pairs (a, b) with a in A and b in B.
a=np.multiply.outer([1, 2, 3], [4, 5, 6])
#array([[ 4,  5,  6],
#       [ 8, 10, 12],
#       [12, 15, 18]])
print(a)
print("++++++++")


#---------At
#Performs unbuffered in place operation on operand ‘a’ for elements specified by ‘indices’. 
a = np.array([1, 2, 3, 4])
b=np.negative.at(a, [0, 1])        #=>array([-1, -2,  3,  4])
                   
c = np.array([1, 2, 3, 4])
d=np.add.at(c, [0, 1, 2, 2], 1)    #=>array([2, 3, 5, 4])

print(a)
print("--------")
print(b)      ##will give None value of a changes
print("--------")
print(c)
print("--------")
print(d)      ##will give None value of c changes
print("++++++++")

#---------add, subtract, multiply
x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
x12=np.add(x1, x2)
x13=x1 + x2                 #The + operator can be used as a shorthand for np.add on ndarrays.
a=np.add(1.0, 4.0)

print(a)
print("--------")
print(x12)     
print("--------")
print(x13)
print("--------")


a=np.subtract(1.0, 4.0)
x12=np.subtract(x1, x2)
x13=x1 - x2                 #Equivalent to x1 - x2 in terms of array broadcasting.

print(a)
print("--------")
print(x12)     
print("--------")
print(x13)
print("++++++++")
#array([[ 0.,  0.,  0.],
#       [ 3.,  3.,  3.],
#       [ 6.,  6.,  6.]])

a=np.multiply(2.0, 4.0)
x12=np.multiply(x1, x2)
x13=x1 * x2                 #The * operator can be used as a shorthand for np.multiply on ndarrays.
#array([[  0.,   1.,   4.],
#       [  0.,   4.,  10.],
#       [  0.,   7.,  16.]])
print(a)
print("--------")
print(x12)     
print("--------")
print(x13)
print("++++++++")



#---------multiply
#Matrix product of two arrays.

a = np.ones([9, 5, 7, 4])
b = np.ones([9, 5, 4, 3])
c1=np.dot(a, b).shape        #(9, 5, 7, 9, 5, 3)
c2=np.matmul(a, b).shape       #(9, 5, 7, 3)
print(c1)
print("--------")
print(c2)
print("--------")

# n is 7, k is 4, m is 3

a = np.array([[1, 0],[0, 1]])
b = np.array([[4, 1],[2, 2]])
c=np.matmul(a, b)                  #array([[4, 1],[2, 2]])
print(c)
print("++++++++")

#---------divide
#Divide arguments element-wise.

x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
a=np.divide(2.0, 4.0)
b=np.divide(x1, x2)
#array([[nan, 1. , 1. ],
#       [inf, 4. , 2.5],
#       [inf, 7. , 4. ]])
c=x1 / x2                     #The / operator can be used as a shorthand for np.divide on ndarrays.
print(a)
print("--------")
print(b)     
print("--------")
print(c)
print("+++++++++")

#---------logaddexp
#Logarithm of the sum of exponentiations of the inputs.
#Calculates log(exp(x1) + exp(x2)). 

prob1 = np.log(1e-50)
prob2 = np.log(2.5e-50)
prob12 = np.logaddexp(prob1, prob2)
print(prob12)               #-113.87649168120691
print("--------")
print(np.exp(prob12))       #3.5000000000000057e-50
print("+++++++++")


#---------logaddexp2
#Logarithm of the sum of exponentiations of the inputs in base-2.
#Calculates log2(2**x1 + 2**x2).

prob1 = np.log2(1e-50)
prob2 = np.log2(2.5e-50)
prob12 = np.logaddexp2(prob1, prob2)
print(prob1, prob2, prob12)        #(-166.09640474436813, -164.77447664948076, -164.28904982231052)
print("--------")
print(2**prob12)                   #3.4999999999999914e-50
print("+++++++++")

#---------numpy.true_divide
#Divide arguments element-wise.
#Equivalent to x1 / x2 in terms of array-broadcasting.

x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)

a=np.divide(2.0, 4.0)
b=np.divide(x1, x2)
c=x1 / x2                   #The / operator can be used as a shorthand for np.divide on ndarrays.

print(a)
print("--------")
print(b)
print("--------")
print(c)
print("+++++++++")

#---------numpy.floor_divide
a=np.floor_divide(7,3)
b=np.floor_divide([1., 2., 3., 4.], 2.5)

x1 = np.array([1., 2., 3., 4.])
c=x1 // 2.5     #The // operator can be used as a shorthand for np.floor_divide on ndarrays.

print(a)
print("--------")
print(b)
print("--------")
print(c)
print("+++++++++")

#---------numpy.negative
#Numerical negative, element-wise.

a=np.negative([1.,-1.])
x1 = np.array(([1., -1.]))
print(a)
print(-x1)                  #The unary - operator can be used as a shorthand for np.negative on ndarrays.
print("+++++++++")

#---------numpy.positive
#Numerical positive, element-wise.

x1 = np.array(([1., -1.]))
a=np.positive(x1)
print(a)
print(+x1)           #The unary + operator can be used as a shorthand for np.positive on ndarrays.
print("+++++++++")



#---------numpy.power
#First array elements raised to powers from second array, element-wise. 
#Also exist float_power method similar

x1 = np.arange(6)
a=np.power(x1, 3)

x2 = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
b=np.power(x1, x2)                          #Raise the bases to different exponents.

x2 = np.array([[1, 2, 3, 3, 2, 1], [1, 2, 3, 3, 2, 1]]) #The effect of broadcasting.
c=np.power(x1, x2)

x2 = np.array([1, 2, 3, 3, 2, 1])
x1 = np.arange(6)
d=x1 ** x2                                #The ** operator can be used as a shorthand for np.power on ndarrays.

x3 = np.array([-1.0, -4.0])
with np.errstate(invalid='ignore'):
    p = np.power(x3, 1.5)                 #Negative values raised to a non-integral value will result in nan (and a warning will be generated).

print(a)
print("--------")
print(b)
print("--------")
print(c)
print("--------")
print(d)
print("--------")
print(p)
print("--------")

e=np.power(x3, 1.5, dtype=complex) #To get complex results, give the argument dtype=complex.
print(e)
print("+++++++++")


#---------numpy.remainder
#Returns the element-wise remainder of division.
#The % operator can be used as a shorthand for np.remainder on ndarrays.

a=np.remainder([4, 7], [2, 3])
b=np.remainder(np.arange(7), 5)

x1 = np.arange(7)
c=x1 % 5
print(a)
print("--------")
print(b)
print("--------")
print(c)
print("+++++++++")

#---------numpy.mod
#exist numpy.fmod similar method
#Returns the element-wise remainder of division.
#The % operator can be used as a shorthand for np.remainder on ndarrays.

a=np.remainder([4, 7], [2, 3])
b=np.remainder(np.arange(7), 5)
x1 = np.arange(7)
c=x1 % 5
print(a)
print("--------")
print(b)
print("--------")
print(c)
print("+++++++++")

#---------numpy.divmod
#Return element-wise quotient and remainder simultaneously.
#The divmod function can be used as a shorthand for np.divmod on ndarrays.

a=np.divmod(np.arange(5), 3)
x = np.arange(5)
b=divmod(x, 3)

print(a)
print("--------")
print(b)
print("+++++++++")

#---------numpy.absolute
#similar method numpy.fabs
#Calculate the absolute value element-wise.

x = np.array([-1.2, 1.2])
a=np.absolute(x)
b=np.absolute(1.2 + 1j)
print(a)
print("--------")
print(b)
print("+++++++++")

#---------numpy.rint
#Round elements of the array to the nearest integer.
a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
b=np.rint(a)
print(b)
print("+++++++++")

#---------numpy.sign
#Returns an element-wise indication of the sign of a number.
a=np.sign([-5., 4.5])
b=np.sign(0)
c=np.sign(5-2j)
print(a)
print("--------")
print(b)
print("--------")
print(c)
print("+++++++++")

#---------numpy.exp2
#Calculate 2**p for all p in the input array.
print(np.exp2([2, 3]))
print("+++++++++")

#---------numpy.log
#Similar function numpy.log2 and numpy.log10
#Natural logarithm, element-wise.

a=np.log([1, np.e, np.e**2, 0])
print(a)
print("+++++++++")

#---------numpy.sqrt
#Return the non-negative square-root of an array, element-wise.
#Similar function with cube numpy.cbrt

a=np.sqrt([1,4,9])
b=np.sqrt([4, -1, -3+4J])
c=np.sqrt([4, -1, np.inf])
print(a)
print("--------")
print(b)
print("--------")
print(c)
print("+++++++++")

#---------numpy.square
#Return the element-wise square of the input.
print(np.square([-1j, 1]))
print("+++++++++")


#---------numpy.reciprocal
#Return the reciprocal of the argument, element-wise.
#Calculates 1/x.

a=np.reciprocal(2.)
b=np.reciprocal([1, 2., 3.33])
print(a)
print("--------")
print(b)
print("+++++++++")

#---------numpy.gcd
#Returns the greatest common divisor of |x1| and |x2|
#Similar function lcm lowest common multipier

a=np.gcd(12, 20)
b=np.gcd.reduce([15, 25, 35])
c=np.gcd(np.arange(6), 20)
print(a)
print("--------")
print(b)
print("--------")
print(c)
print("+++++++++")



#---------numpy.greater
#similar functions numpy.greater_equal, numpy.less, numpy.less_equal
#Return the truth value of (x1 > x2) element-wise.
#The > operator can be used as a shorthand for np.greater on ndarrays.

a=np.greater([4,2],[2,2])

b= np.array([4, 2])
c= np.array([2, 2])
e=b > c

print(a)
print("--------")
print(e)
print("+++++++++")

#---------numpy.not_equal
#Similar functions numpy.equal
#Return (x1 != x2) element-wise.
#The != operator can be used as a shorthand for np.not_equal on ndarrays.

a=np.not_equal([1.,2.], [1., 3.])
b=np.not_equal([1, 2], [[1, 3],[1, 4]])

print(a)
print("--------")
print(b)
print("--------")

a = np.array([1., 2.])
b = np.array([1., 3.])
c=a != b
print(c)
print("+++++++++")


#---------numpy.logical_and
#Similar functions logical_or, logical_xor, logical_not
#Compute the truth value of x1 AND x2 element-wise.
#The & operator can be used as a shorthand for np.logical_and on boolean ndarrays.

a=np.logical_and(True, False)
b=np.logical_and([True, False], [False, False])
x = np.arange(5)
c=np.logical_and(x>1, x<4)

x = np.array([True, False])
y = np.array([False, False])
e=x & y

print(a)
print("--------")
print(b)
print("--------")
print(c)
print("--------")
print(e)
print("+++++++++")

#---------numpy.maximum
#Element-wise maximum of array elements.
#Similar function minimum, fmax, fmin

a=np.maximum([2, 3, 4], [1, 5, 2])
b=np.maximum(np.eye(2), [0.5, 2]) # broadcasting
c=np.maximum([np.nan, 0, np.nan], [0, np.nan, np.nan])
e=np.maximum(np.Inf, 1)
print(a)
print("--------")
print(b)
print("--------")
print(c)
print("--------")
print(e)
print("+++++++++")

#---------numpy.isfinite
#Test element-wise for finiteness (not infinity and not Not a Number).
#The result is returned as a boolean array.

print(np.isfinite(1))
print("--------")
print(np.isfinite(0))
print("--------")
print(np.isfinite(np.nan))
print("--------")
print(np.isfinite(np.inf))
print("--------")
print(np.isfinite(np.NINF))
print("--------")
print(np.isfinite([np.log(-1.),1.,np.log(0)]))
print("+++++++++")



#---------numpy.isinf
#Test element-wise for positive or negative infinity.
#Returns a boolean array of the same shape as x, True where x == +/-inf, otherwise False.

print(np.isinf(np.inf))
print("--------")
print(np.isinf(np.nan))
print("--------")
print(np.isinf(np.NINF))
print("--------")
print(np.isinf([np.inf, -np.inf, 1.0, np.nan]))
print("--------")

x = np.array([-np.inf, 0., np.inf])
y = np.array([2, 2, 2])
print(np.isinf(x, y))
print("+++++++++")

#---------numpy.isnan
#Test element-wise for NaN and return result as a boolean array.

print(np.isnan(np.nan))
print("--------")
print(np.isnan(np.inf))
print("--------")
print(np.isnan([np.log(-1.),1.,np.log(0)]))
print("+++++++++")

#---------numpy.spacing
#Return the distance between x and the nearest adjacent number.
print(np.spacing(1) == np.finfo(np.float64).eps)
print("+++++++++")

#---------numpy.spacing
#Compute the absolute values element-wise.
#This function returns the absolute values (positive magnitude) of the data in x.

print(np.fabs(-1))
print("--------")
print(np.fabs([-1.2, 1.2]))
print("+++++++++")

#---------Other Functions
#numpy.heaviside
#numpy.conj
#numpy.conjugate
#numpy.exp
#numpy.expm1
#numpy.log1p


'''
#---------Trigonometric functions
All trigonometric functions use radians when an angle is called for. The ratio of degrees to radians is 

sin(x, /[, out, where, casting, order, ...])
Trigonometric sine, element-wise.

cos(x, /[, out, where, casting, order, ...])
Cosine element-wise.

tan(x, /[, out, where, casting, order, ...])
Compute tangent element-wise.

arcsin(x, /[, out, where, casting, order, ...])
Inverse sine, element-wise.

arccos(x, /[, out, where, casting, order, ...])
Trigonometric inverse cosine, element-wise.

arctan(x, /[, out, where, casting, order, ...])
Trigonometric inverse tangent, element-wise.

arctan2(x1, x2, /[, out, where, casting, ...])
Element-wise arc tangent of x1/x2 choosing the quadrant correctly.

hypot(x1, x2, /[, out, where, casting, ...])
Given the "legs" of a right triangle, return its hypotenuse.

sinh(x, /[, out, where, casting, order, ...])
Hyperbolic sine, element-wise.

cosh(x, /[, out, where, casting, order, ...])
Hyperbolic cosine, element-wise.

tanh(x, /[, out, where, casting, order, ...])
Compute hyperbolic tangent element-wise.

arcsinh(x, /[, out, where, casting, order, ...])
Inverse hyperbolic sine element-wise.

arccosh(x, /[, out, where, casting, order, ...])
Inverse hyperbolic cosine, element-wise.

arctanh(x, /[, out, where, casting, order, ...])
Inverse hyperbolic tangent element-wise.

degrees(x, /[, out, where, casting, order, ...])
Convert angles from radians to degrees.

radians(x, /[, out, where, casting, order, ...])
Convert angles from degrees to radians.

deg2rad(x, /[, out, where, casting, order, ...])
Convert angles from degrees to radians.

rad2deg(x, /[, out, where, casting, order, ...])
Convert angles from radians to degrees.
Bit-twiddling functions
These function all require integer arguments and they manipulate the bit-pattern of those arguments.

bitwise_and(x1, x2, /[, out, where, ...])
Compute the bit-wise AND of two arrays element-wise.

bitwise_or(x1, x2, /[, out, where, casting, ...])
Compute the bit-wise OR of two arrays element-wise.

bitwise_xor(x1, x2, /[, out, where, ...])
Compute the bit-wise XOR of two arrays element-wise.

invert(x, /[, out, where, casting, order, ...])
Compute bit-wise inversion, or bit-wise NOT, element-wise.

left_shift(x1, x2, /[, out, where, casting, ...])
Shift the bits of an integer to the left.

right_shift(x1, x2, /[, out, where, ...])
Shift the bits of an integer to the right.

#---------Floating functions

signbit(x, /[, out, where, casting, order, ...])
Returns element-wise True where signbit is set (less than zero).

copysign(x1, x2, /[, out, where, casting, ...])
Change the sign of x1 to that of x2, element-wise.

nextafter(x1, x2, /[, out, where, casting, ...])
Return the next floating-point value after x1 towards x2, element-wise.

modf(x[, out1, out2], / [[, out, where, ...])
Return the fractional and integral parts of an array, element-wise.

ldexp(x1, x2, /[, out, where, casting, ...])
Returns x1 * 2**x2, element-wise.

frexp(x[, out1, out2], / [[, out, where, ...])
Decompose the elements of x into mantissa and twos exponent.

fmod(x1, x2, /[, out, where, casting, ...])
Returns the element-wise remainder of division.

floor(x, /[, out, where, casting, order, ...])
Return the floor of the input, element-wise.

ceil(x, /[, out, where, casting, order, ...])
Return the ceiling of the input, element-wise.

trunc(x, /[, out, where, casting, order, ...])
Return the truncated value of the input, element-wise.
'''



