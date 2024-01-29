import numpy as np

#Functions that operate element by element on whole arrays.




#---------Reduce
#Reduces array’s dimension by one, by applying ufunc along one axis.
np.multiply.reduce([2,3,5])
X = np.arange(8).reshape((2,2,2))
np.add.reduce(X, 0)
np.add.reduce(X)
np.add.reduce(X, 1)
np.add.reduce(X, 2)
np.add.reduce([10], initial=5)  #The value with which to start the reduction.


#---------Accumulate
#Accumulate the result of applying the operator to all elements.
np.add.accumulate([2, 3, 5])
np.multiply.accumulate([2, 3, 5])


#---------ReduceAt
#Performs a (local) reduce with specified slices over a single axis.
np.add.reduceat(np.arange(8),[0,4, 1,5, 2,6, 3,7])[::2]     #=>array([ 6, 10, 14, 18])
#To take the running sum of four successive values:

x = np.linspace(0, 15, 16).reshape(4,4)
print(x)
#array([[ 0.,   1.,   2.,   3.],
#       [ 4.,   5.,   6.,   7.],
#       [ 8.,   9.,  10.,  11.],
#       [12.,  13.,  14.,  15.]])

np.add.reduceat(x, [0, 3, 1, 2, 0])


#---------Outer
#Apply the ufunc op to all pairs (a, b) with a in A and b in B.
np.multiply.outer([1, 2, 3], [4, 5, 6])
#array([[ 4,  5,  6],
#       [ 8, 10, 12],
#       [12, 15, 18]])


#---------At
#Performs unbuffered in place operation on operand ‘a’ for elements specified by ‘indices’. 
a = np.array([1, 2, 3, 4])
np.negative.at(a, [0, 1])
print(a)                    #=>array([-1, -2,  3,  4])

a = np.array([1, 2, 3, 4])
np.add.at(a, [0, 1, 2, 2], 1)
print(a)                    #=>array([2, 3, 5, 4])




#---------add
print(np.add(1.0, 4.0))
x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
x12=np.add(x1, x2)
#The + operator can be used as a shorthand for np.add on ndarrays.
x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
x12=x1 + x2


#---------subtract
np.subtract(1.0, 4.0)
-3.0

x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
np.subtract(x1, x2)
array([[ 0.,  0.,  0.],
       [ 3.,  3.,  3.],
       [ 6.,  6.,  6.]])

x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
x1 - x2
array([[0., 0., 0.],
       [3., 3., 3.],
       [6., 6., 6.]])

#Equivalent to x1 - x2 in terms of array broadcasting.



#---------multiply
np.multiply(2.0, 4.0)
8.0
x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
np.multiply(x1, x2)
array([[  0.,   1.,   4.],
       [  0.,   4.,  10.],
       [  0.,   7.,  16.]])
The * operator can be used as a shorthand for np.multiply on ndarrays.

x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
x1 * x2
array([[  0.,   1.,   4.],
       [  0.,   4.,  10.],
       [  0.,   7.,  16.]])


#---------multiply
Matrix product of two arrays.

a = np.ones([9, 5, 7, 4])
c = np.ones([9, 5, 4, 3])
np.dot(a, c).shape
(9, 5, 7, 9, 5, 3)
np.matmul(a, c).shape
(9, 5, 7, 3)
# n is 7, k is 4, m is 3

a = np.array([[1, 0],
              [0, 1]])
b = np.array([[4, 1],
              [2, 2]])
np.matmul(a, b)
array([[4, 1],
       [2, 2]])





#---------divide

Divide arguments element-wise.

np.divide(2.0, 4.0)
0.5
x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
np.divide(x1, x2)
array([[nan, 1. , 1. ],
       [inf, 4. , 2.5],
       [inf, 7. , 4. ]])
The / operator can be used as a shorthand for np.divide on ndarrays.

x1 = np.arange(9.0).reshape((3, 3))
x2 = 2 * np.ones(3)
x1 / x2
array([[0. , 0.5, 1. ],
       [1.5, 2. , 2.5],
       [3. , 3.5, 4. ]])

#---------logaddexp
Logarithm of the sum of exponentiations of the inputs.

Calculates log(exp(x1) + exp(x2)). 

prob1 = np.log(1e-50)
prob2 = np.log(2.5e-50)
prob12 = np.logaddexp(prob1, prob2)
prob12
-113.87649168120691
np.exp(prob12)
3.5000000000000057e-50



#---------logaddexp2

numpy.logaddexp2
Logarithm of the sum of exponentiations of the inputs in base-2.

Calculates log2(2**x1 + 2**x2).

prob1 = np.log2(1e-50)
prob2 = np.log2(2.5e-50)
prob12 = np.logaddexp2(prob1, prob2)
prob1, prob2, prob12
(-166.09640474436813, -164.77447664948076, -164.28904982231052)
2**prob12
3.4999999999999914e-50


#---------numpy.true_divide
#Divide arguments element-wise.

#Equivalent to x1 / x2 in terms of array-broadcasting.



np.divide(2.0, 4.0)
0.5
x1 = np.arange(9.0).reshape((3, 3))
x2 = np.arange(3.0)
np.divide(x1, x2)
array([[nan, 1. , 1. ],
       [inf, 4. , 2.5],
       [inf, 7. , 4. ]])
The / operator can be used as a shorthand for np.divide on ndarrays.

x1 = np.arange(9.0).reshape((3, 3))
x2 = 2 * np.ones(3)
x1 / x2
array([[0. , 0.5, 1. ],
       [1.5, 2. , 2.5],
       [3. , 3.5, 4. ]])



#---------numpy.floor_divide


np.floor_divide(7,3)
2
np.floor_divide([1., 2., 3., 4.], 2.5)
array([ 0.,  0.,  1.,  1.])
The // operator can be used as a shorthand for np.floor_divide on ndarrays.

x1 = np.array([1., 2., 3., 4.])
x1 // 2.5
array([0., 0., 1., 1.])



#---------numpy.negative

Numerical negative, element-wise.

np.negative([1.,-1.])
array([-1.,  1.])
The unary - operator can be used as a shorthand for np.negative on ndarrays.

x1 = np.array(([1., -1.]))
-x1
array([-1.,  1.])






#---------numpy.positive
Numerical positive, element-wise.

x1 = np.array(([1., -1.]))
np.positive(x1)
array([ 1., -1.])
The unary + operator can be used as a shorthand for np.positive on ndarrays.

x1 = np.array(([1., -1.]))
+x1
array([ 1., -1.])



#---------numpy.power

First array elements raised to powers from second array, element-wise.
Also exist float_power method similar

x1 = np.arange(6)
x1
[0, 1, 2, 3, 4, 5]
np.power(x1, 3)
array([  0,   1,   8,  27,  64, 125])
Raise the bases to different exponents.

x2 = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
np.power(x1, x2)
array([  0.,   1.,   8.,  27.,  16.,   5.])
The effect of broadcasting.

x2 = np.array([[1, 2, 3, 3, 2, 1], [1, 2, 3, 3, 2, 1]])
x2
array([[1, 2, 3, 3, 2, 1],
       [1, 2, 3, 3, 2, 1]])
np.power(x1, x2)
array([[ 0,  1,  8, 27, 16,  5],
       [ 0,  1,  8, 27, 16,  5]])
The ** operator can be used as a shorthand for np.power on ndarrays.

x2 = np.array([1, 2, 3, 3, 2, 1])
x1 = np.arange(6)
x1 ** x2
array([ 0,  1,  8, 27, 16,  5])
Negative values raised to a non-integral value will result in nan (and a warning will be generated).

x3 = np.array([-1.0, -4.0])
with np.errstate(invalid='ignore'):
    p = np.power(x3, 1.5)

p
array([nan, nan])
To get complex results, give the argument dtype=complex.

np.power(x3, 1.5, dtype=complex)
array([-1.83697020e-16-1.j, -1.46957616e-15-8.j])


#---------numpy.remainder

Returns the element-wise remainder of division.

np.remainder([4, 7], [2, 3])
array([0, 1])
np.remainder(np.arange(7), 5)
array([0, 1, 2, 3, 4, 0, 1])
The % operator can be used as a shorthand for np.remainder on ndarrays.

x1 = np.arange(7)
x1 % 5
array([0, 1, 2, 3, 4, 0, 1])


#---------numpy.mod
exist numpy.fmod similar method
Returns the element-wise remainder of division.

np.remainder([4, 7], [2, 3])
array([0, 1])
np.remainder(np.arange(7), 5)
array([0, 1, 2, 3, 4, 0, 1])
The % operator can be used as a shorthand for np.remainder on ndarrays.

x1 = np.arange(7)
x1 % 5
array([0, 1, 2, 3, 4, 0, 1])



#---------numpy.divmod
Return element-wise quotient and remainder simultaneously.

np.divmod(np.arange(5), 3)
(array([0, 0, 0, 1, 1]), array([0, 1, 2, 0, 1]))
The divmod function can be used as a shorthand for np.divmod on ndarrays.

x = np.arange(5)
divmod(x, 3)
(array([0, 0, 0, 1, 1]), array([0, 1, 2, 0, 1]))



#---------numpy.absolute
#similar method numpy.fabs
Calculate the absolute value element-wise.

x = np.array([-1.2, 1.2])
np.absolute(x)
array([ 1.2,  1.2])
np.absolute(1.2 + 1j)
1.5620499351813308

#---------numpy.rint
Round elements of the array to the nearest integer.
a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
np.rint(a)
array([-2., -2., -0.,  0.,  2.,  2.,  2.])


#---------numpy.sign
Returns an element-wise indication of the sign of a number.

np.sign([-5., 4.5])
array([-1.,  1.])
np.sign(0)
0
np.sign(5-2j)
(1+0j)




#numpy.heaviside
#numpy.conj
#numpy.conjugate
#numpy.exp
numpy.expm1
numpy.log1p


#---------numpy.exp2

Calculate 2**p for all p in the input array.
np.exp2([2, 3])
array([ 4.,  8.])



#---------numpy.log
similar function numpy.log2 and numpy.log10
Natural logarithm, element-wise.


np.log([1, np.e, np.e**2, 0])
array([  0.,   1.,   2., -Inf])

#---------numpy.sqrt
Return the non-negative square-root of an array, element-wise.
Similar function with cube numpy.cbrt

np.sqrt([1,4,9])
array([ 1.,  2.,  3.])
np.sqrt([4, -1, -3+4J])
array([ 2.+0.j,  0.+1.j,  1.+2.j])
np.sqrt([4, -1, np.inf])
array([ 2., nan, inf])


#---------numpy.square

Return the element-wise square of the input.

np.square([-1j, 1])
array([-1.-0.j,  1.+0.j])


#---------numpy.reciprocal
Return the reciprocal of the argument, element-wise.

Calculates 1/x.

np.reciprocal(2.)
0.5
np.reciprocal([1, 2., 3.33])
array([ 1.       ,  0.5      ,  0.3003003])


#---------numpy.gcd
Returns the greatest common divisor of |x1| and |x2|
Similar function lcm lowest common multipier


np.gcd(12, 20)
4
np.gcd.reduce([15, 25, 35])
5
np.gcd(np.arange(6), 20)
array([20,  1,  2,  1,  4,  5])



#---------numpy.greater
similar functions numpy.greater_equal, numpy.less, numpy.less_equal
Return the truth value of (x1 > x2) element-wise.

np.greater([4,2],[2,2])
array([ True, False])
The > operator can be used as a shorthand for np.greater on ndarrays.

a = np.array([4, 2])
b = np.array([2, 2])
a > b
array([ True, False])

#---------numpy.not_equal
Similar functions numpy.equal

Return (x1 != x2) element-wise.

np.not_equal([1.,2.], [1., 3.])
array([False,  True])
np.not_equal([1, 2], [[1, 3],[1, 4]])
array([[False,  True],
       [False,  True]])
The != operator can be used as a shorthand for np.not_equal on ndarrays.

a = np.array([1., 2.])
b = np.array([1., 3.])
a != b
array([False,  True])

#---------numpy.logical_and

Similar functions logical_or, logical_xor, logical_not

Compute the truth value of x1 AND x2 element-wise.

np.logical_and(True, False)
False
np.logical_and([True, False], [False, False])
array([False, False])
x = np.arange(5)
np.logical_and(x>1, x<4)
array([False, False,  True,  True, False])
The & operator can be used as a shorthand for np.logical_and on boolean ndarrays.

a = np.array([True, False])
b = np.array([False, False])
a & b
array([False, False])





#---------numpy.maximum
Element-wise maximum of array elements.
Similar function minimum, fmax, fmin

np.maximum([2, 3, 4], [1, 5, 2])
array([2, 5, 4])
np.maximum(np.eye(2), [0.5, 2]) # broadcasting
array([[ 1. ,  2. ],
       [ 0.5,  2. ]])
np.maximum([np.nan, 0, np.nan], [0, np.nan, np.nan])
array([nan, nan, nan])
np.maximum(np.Inf, 1)
inf


#---------numpy.isfinite

Test element-wise for finiteness (not infinity and not Not a Number).

The result is returned as a boolean array.

np.isfinite(1)
True
np.isfinite(0)
True
np.isfinite(np.nan)
False
np.isfinite(np.inf)
False
np.isfinite(np.NINF)
False
np.isfinite([np.log(-1.),1.,np.log(0)])
array([False,  True, False])



#---------numpy.isinf
Test element-wise for positive or negative infinity.

Returns a boolean array of the same shape as x, True where x == +/-inf, otherwise False.

np.isinf(np.inf)
True
np.isinf(np.nan)
False
np.isinf(np.NINF)
True
np.isinf([np.inf, -np.inf, 1.0, np.nan])
array([ True,  True, False, False])
x = np.array([-np.inf, 0., np.inf])
y = np.array([2, 2, 2])
np.isinf(x, y)
array([1, 0, 1])
y
array([1, 0, 1])

#---------numpy.isnan
Test element-wise for NaN and return result as a boolean array.

np.isnan(np.nan)
True
np.isnan(np.inf)
False
np.isnan([np.log(-1.),1.,np.log(0)])
array([ True, False, False])


#---------numpy.spacing

Return the distance between x and the nearest adjacent number.

np.spacing(1) == np.finfo(np.float64).eps

#---------numpy.spacing
Compute the absolute values element-wise.

This function returns the absolute values (positive magnitude) of the data in x.
np.fabs(-1)
1.0
np.fabs([-1.2, 1.2])
array([ 1.2,  1.2])





Trigonometric functions
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




Floating functions

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




