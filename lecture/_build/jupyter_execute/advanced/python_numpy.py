#!/usr/bin/env python
# coding: utf-8

# # Introduction to NumPy

# **Disclaimer:** Most of the content in this notebook is coming from [www.scipy-lectures.org](http://www.scipy-lectures.org/intro/index.html)

# # NumPy
# 
# [NumPy](http://www.numpy.org/) is **the** fundamental package for scientific computing with Python. It is the basic building block of most data analysis in Python and contains highly optimized routines for creating and manipulating arrays.
# 
# ### Everything revolves around numpy arrays
# * **`Scipy`** adds a bunch of useful science and engineering routines that operate on numpy arrays. E.g. signal processing, statistical distributions, image analysis, etc.
# * **`pandas`** adds powerful methods for manipulating numpy arrays. Like data frames in R - but typically faster.
# * **`scikit-learn`** supports state-of-the-art machine learning over numpy arrays. Inputs and outputs of virtually all functions are numpy arrays.
# * If you want many more short exercises than the ones in this notebook - you can find 100 of them [here](http://www.labri.fr/perso/nrougier/teaching/numpy.100/)

# ## NumPy arrays vs Python arrays
# 
# NumPy arrays look very similar to Pyhon arrays.

# In[1]:


import numpy as np
a = np.array([0, 1, 2, 3])
a


# **So, why is this useful?** NumPy arrays are memory-efficient containers that provide fast numerical operations. We can show this very quickly by running a simple computation on the same two arrays.

# In[2]:


L = range(1000)

# Computing the power of the first 1000 numbers with Python arrays
get_ipython().run_line_magic('timeit', '[i**2 for i in L]')


# In[3]:


a = np.array(L)

# Computing the power of the first 1000 numbers with Numpy arrays
get_ipython().run_line_magic('timeit', 'a**2')


# # Creating arrays
# 
# ## Manual construction of arrays
# 
# You can create NumPy arrays manually almost in the same way as in Python in general.
# 
# ### 1-D

# In[4]:


a = np.array([0, 1, 2, 3])
a


# In[5]:


a.shape


# ### 2-D, 3-D, ...

# In[6]:


b = np.array([[0, 1, 2], [3, 4, 5]])    # 2 x 3 array
b


# In[7]:


b.shape


# In[8]:


c = np.array([[[1], [2]], [[3], [4]]])
c


# In[9]:


c.shape


# ## Functions for creating arrays
# 
# In practice, we rarely enter items one by one. Therefore, NumPy offers many different helper functions.
# 
# ### Evenly spaced

# In[10]:


a = np.arange(10) # 0 .. n-1  (!)
a


# In[11]:


b = np.arange(1, 9, 2) # start, end (exclusive), step
b


# ### ... or by number of points

# In[12]:


c = np.linspace(0, 1, 6)   # start, end, num-points
c


# In[13]:


d = np.linspace(0, 1, 5, endpoint=False)
d


# ### Common arrays

# In[14]:


a = np.ones((3, 3))  # reminder: (3, 3) is a tuple
a


# In[15]:


b = np.zeros((2, 2))
b


# In[16]:


c = np.eye(3)
c


# In[17]:


d = np.diag(np.array([1, 2, 3, 4]))
d


# ### `np.random`: random numbers (Mersenne Twister PRNG)

# In[18]:


a = np.random.rand(4)       # uniform in [0, 1]
a


# In[19]:


b = np.random.randn(4)      # Gaussian
b


# In[20]:


np.random.seed(1234)        # Setting the random seed


# # Basic data types
# 
# You may have noticed that, in some instances, array elements are displayed with a trailing dot (e.g. ``2.`` vs ``2``). This is due to a difference in the data-type used:

# In[21]:


a = np.array([1, 2, 3])
a.dtype


# In[22]:


b = np.array([1., 2., 3.])
b.dtype


# Different data-types allow us to store data more compactly in memory, but most of the time we simply work with floating point numbers. Note that, in the example above, NumPy auto-detects the data-type from the input.
# 
# You can explicitly specify which data-type you want:

# In[23]:


c = np.array([1, 2, 3], dtype=float)
c.dtype


# The **default** data type is floating point:

# In[24]:


a = np.ones((3, 3))
a.dtype


# There are also other types:

# In[25]:


# Complex
d = np.array([1+2j, 3+4j, 5+6*1j])
d.dtype


# In[26]:


# Bool
e = np.array([True, False, False, True])
e.dtype


# In[27]:


# Strings
f = np.array(['Bonjour', 'Hello', 'Hallo',])
f.dtype     # <--- strings containing max. 7 letters


# And much more...
# * ``int32``
# * ``int64``
# * ``uint32``
# * ``uint64``

# # Indexing and slicing
# 
# The items of an array can be accessed and assigned to the same way as other Python sequences (e.g. lists):

# In[28]:


a = np.arange(10)
a


# In[29]:


a[0], a[2], a[-1]


# **Warning**: Indices begin at 0, like other Python sequences (and C/C++). In contrast, in Fortran or Matlab, indices begin at 1.

# The usual python idiom for reversing a sequence is supported:

# In[30]:


a[::-1]


# For multidimensional arrays, indexes are tuples of integers:

# In[31]:


a = np.diag(np.arange(3))
a


# In[32]:


a[1, 1]


# In[33]:


a[2, 1] = 10 # third line, second column
a


# In[34]:


a[1]


# ### Note
# 
# * In 2D, the first dimension corresponds to **rows**, the second to **columns**.
# * For multidimensional ``a``, ``a[0]`` is interpreted by taking all elements in the unspecified dimensions.

# ## Slicing: Arrays, like other Python sequences can also be sliced

# In[35]:


a = np.arange(10)
a


# In[36]:


a[2:9:3] # [start:end:step]


# Note that the last index is not included!

# In[37]:


a[:4]


# All three slice components are not required: by default, `start` is 0,
# `end` is the last and `step` is 1:

# In[38]:


a[1:3]


# In[39]:


a[::2]


# In[40]:


a[3:]


# A small illustrated summary of NumPy indexing and slicing...
# 
# <img src="http://www.scipy-lectures.org/_images/numpy_indexing.png" width=60%>

# You can also combine assignment and slicing:

# In[41]:


a = np.arange(10)
a[5:] = 10
a


# In[42]:


b = np.arange(5)
a[5:] = b[::-1]
a


# # Fancy indexing
# 
# NumPy arrays can be indexed with slices, but also with boolean or integer arrays (**masks**). This method is called *fancy indexing*. It creates **copies not views**.
# 
# ## Using boolean masks

# In[43]:


np.random.seed(3)
a = np.random.randint(0, 21, 15)
a


# In[44]:


(a % 3 == 0)


# In[45]:


mask = (a % 3 == 0)
extract_from_a = a[mask] # or,  a[a%3==0]
extract_from_a           # extract a sub-array with the mask


# Indexing with a mask can be very useful to assign a new value to a sub-array:

# In[46]:


a[a % 3 == 0] = -1
a


# ## Indexing with an array of integers

# In[47]:


a = np.arange(0, 100, 10)
a


# Indexing can be done with an array of integers, where the same index is repeated several time:

# In[48]:


a[[2, 3, 2, 4, 2]]  # note: [2, 3, 2, 4, 2] is a Python list


# New values can be assigned with this kind of indexing:

# In[49]:


a[[9, 7]] = -100
a


# The image below illustrates various fancy indexing applications:
# 
# <img src="http://www.scipy-lectures.org/_images/numpy_fancy_indexing.png" width=60%>

# # Elementwise operations
# 
# NumPy provides many elementwise operations that are much quicker than comparable list comprehension in plain Python.
# 
# ## Basic operations
# 
# With scalars:

# In[50]:


a = np.array([1, 2, 3, 4])
a + 1


# In[51]:


2**a


# All arithmetic operates elementwise:

# In[52]:


b = np.ones(4) + 1
a - b


# In[53]:


a * b


# In[54]:


j = np.arange(5)
2**(j + 1) - j


# ### Array multiplication is not matrix multiplication

# In[55]:


c = np.ones((3, 3))
c * c                   # NOT matrix multiplication!


# In[56]:


# Matrix multiplication
c.dot(c)


# ## Other operations
# 
# ### Comparisons

# In[57]:


a = np.array([1, 2, 3, 4])
b = np.array([4, 2, 2, 4])
a == b


# In[58]:


a > b


# Array-wise comparisons:

# In[59]:


a = np.array([1, 2, 3, 4])
b = np.array([4, 2, 2, 4])
c = np.array([1, 2, 3, 4])
np.array_equal(a, b)


# In[60]:


np.array_equal(a, c)


# ### Logical operations

# In[61]:


a = np.array([1, 1, 0, 0], dtype=bool)
b = np.array([1, 0, 1, 0], dtype=bool)
np.logical_or(a, b)


# In[62]:


np.logical_and(a, b)


# ### Transcendental functions

# In[63]:


a = np.arange(5)
np.sin(a)


# In[64]:


np.log(a)


# In[65]:


np.exp(a)


# ### Shape mismatches

# In[66]:


a = np.arange(4)


# In[67]:


# NBVAL_SKIP
a + np.array([1, 2])


# *Broadcasting?* We'll return to that later.

# ### Transposition

# In[68]:


a = np.triu(np.ones((3, 3)), 1)
a


# In[69]:


a.T


# ### The transposition is a view
# 
# As a result, the following code **is wrong** and will **not make a matrix symmetric**:
# 
#     >>> a += a.T
# 
# It will work for small arrays (because of buffering) but fail for large one, in unpredictable ways.

# # Basic reductions
# 
# NumPy offers many quick functions to compute things like sum, mean, max etc.
# 
# ## Computing sums

# In[70]:


x = np.array([1, 2, 3, 4])
np.sum(x)


# Note: Certain NumPy functions can be also written at the end of an Numpy array.

# In[71]:


x.sum()


# Sum by rows and by columns:
# 
# <img src="http://www.scipy-lectures.org/_images/reductions.png" width=20%>

# In[72]:


x = np.array([[1, 1], [2, 2]])
x


# In[73]:


x.sum(axis=0)   # columns (first dimension)


# In[74]:


x.sum(axis=1)   # rows (second dimension)


# ## Other reductions
# 
# Like, `mean`, `std`, `cumsum` etc. works the same way (and take ``axis=``).

# ### Extrema

# In[75]:


x = np.array([1, 3, 2])
x.min()


# In[76]:


x.max()


# In[77]:


x.argmin()  # index of minimum


# In[78]:


x.argmax()  # index of maximum


# ### Logical operations

# In[79]:


np.all([True, True, False])


# In[80]:


np.any([True, True, False])


# Can be used for array comparisons:

# In[81]:


a = np.zeros((100, 100))
np.any(a != 0)


# In[82]:


np.all(a == a)


# In[83]:


a = np.array([1, 2, 3, 2])
b = np.array([2, 2, 3, 2])
c = np.array([6, 4, 4, 5])
((a <= b) & (b <= c)).all()


# ### Statistics

# In[84]:


x = np.array([1, 2, 3, 1])
y = np.array([[1, 2, 3], [5, 6, 1]])
x.mean()


# In[85]:


np.median(x)


# In[86]:


np.median(y, axis=-1) # last axis


# In[87]:


x.std()          # full population standard dev.


# ... and many more (best to learn as you go).

# # Broadcasting
# 
# * Basic operations on ``numpy`` arrays (addition, etc.) are elementwise
# 
# * This works on arrays of the same size. ***Nevertheless***, It's also possible to do operations on arrays of different sizes if *NumPy* can transform these arrays so that they all have the same size: this conversion is called **broadcasting**.
# 
# The image below gives an example of broadcasting:
# 
# <img src="http://www.scipy-lectures.org/_images/numpy_broadcasting.png" width=75%>

# Let's verify this:

# In[88]:


a = np.tile(np.arange(0, 40, 10), (3, 1)).T
a


# In[89]:


b = np.array([0, 1, 2])
a + b


# We have already used broadcasting without knowing it!

# In[90]:


a = np.ones((4, 5))
a[0] = 2  # we assign an array of dimension 0 to an array of dimension 1
a


# An useful trick:

# In[91]:


a = np.arange(0, 40, 10)
a.shape


# In[92]:


a = a[:, np.newaxis]  # adds a new axis -> 2D array
a.shape
a


# In[93]:


a + b


# Broadcasting seems a bit magical, but it is actually quite natural to use it when we want to solve a problem whose output data is an array with more dimensions than input data.

# A lot of grid-based or network-based problems can also use broadcasting. For instance, if we want to compute the distance from the origin of points on a 10x10 grid, we can do:

# In[94]:


x, y = np.arange(5), np.arange(5)[:, None]
distance = np.sqrt(x ** 2 + y ** 2)
distance


# Or in color:

# # Array shape manipulation
# 
# Sometimes your arrays don't have the right shape. Also for this, NumPy has many solutions.
# 
# ## Flattening

# In[95]:


a = np.array([[1, 2, 3], [4, 5, 6]])
a


# In[96]:


a.ravel()


# In[97]:


a.T


# In[98]:


a.T.ravel()


# Higher dimensions: last dimensions ravel out "first".

# ## Reshaping
# 
# The inverse operation to flattening:

# In[99]:


a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
a


# In[100]:


a.reshape(6, 2)


# Or,

# In[101]:


a.reshape((6, -1))    # unspecified (-1) value is inferred


# ## Adding a dimension
# 
# Indexing with the ``np.newaxis`` or ``None`` object allows us to add an axis to an array (you have seen this already above in the broadcasting section):

# In[102]:


z = np.array([1, 2, 3])
z


# In[103]:


z[:, np.newaxis]


# In[104]:


z[np.newaxis, :]


# ## Dimension shuffling

# In[105]:


a = np.arange(4*3*2).reshape(4, 3, 2)
a


# In[106]:


a.shape


# In[107]:


b = a.transpose(1, 2, 0)
b


# In[108]:


b.shape


# ## Resizing
# 
# Size of an array can be changed with ``ndarray.resize``:

# In[109]:


a = np.arange(4)
a.resize((8,))
a


# # Sorting data
# 
# Sorting along an axis:

# In[110]:


a = np.array([[4, 3, 5], [1, 2, 1]])
b = np.sort(a, axis=1)
b


# **Important**: Note that the code above sorts each row separately!

# In-place sort:

# In[111]:


a.sort(axis=1)
a


# Sorting with fancy indexing:

# In[112]:


a = np.array([4, 3, 1, 2])
j = np.argsort(a)
j


# In[113]:


a[j]


# Finding minima and maxima:

# In[114]:


a = np.array([4, 3, 1, 2])
j_max = np.argmax(a)
j_min = np.argmin(a)
j_max, j_min


# # `npy` - NumPy's own data format
# 
# NumPy has its own binary format, not portable but with efficient I/O:

# In[115]:


data = np.ones((3, 3))
np.save('pop.npy', data)
data3 = np.load('pop.npy')


# # Summary - What do you need to know to get started?
# 
# * Know how to create arrays : ``array``, ``arange``, ``ones``, ``zeros``.
# * Know the shape of the array with ``array.shape``, then use slicing to obtain different views of the array: ``array[::2]``, etc. Adjust the shape of the array using ``reshape`` or flatten it with ``ravel``.
# * Obtain a subset of the elements of an array and/or modify their values with masks  
#      ``a[a < 0] = 0``
# * Know miscellaneous operations on arrays, such as finding the mean or max (``array.max()``, ``array.mean()``).
# * For advanced use: master the indexing with arrays of integers, as well as broadcasting.
