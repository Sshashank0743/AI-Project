#!/usr/bin/env python
# coding: utf-8

# # 1.1. Speed comparision between Numpy and Python Lists

# In[1]:


import time
import numpy as np

size_of_vec = 1000000

def pure_python_version():                                                # This function will return the time for python calculation
    time_python = time.time()                                             # Start time before operation
    my_list1 = range(size_of_vec)                                         # Creating a list with 1000000 values
    my_list2 = range(size_of_vec)
    sum_list = [my_list1[i] + my_list2[i] for i in range(len(my_list1))]  # Calculating the sum
    return time.time() - time_python                                      # Return Current time - start time

def numpy_version():                                                      # This function will return the time for numpy calculation
    time_numpy = time.time()                                              # Start time before operation
    my_arr1 = np.arange(size_of_vec)                                      # Creating a numpy array of 1000000 values
    my_arr2 = np.arange(size_of_vec)
    sum_array = my_arr1 + my_arr2                                         # Calculate the sum
    return time.time() - time_numpy                                       # Return current time - start time


python_time = pure_python_version()                                       # Time taken for Python expression
numpy_time = numpy_version()                                              # Time taken for numpy operation
print("Pure Python version {:0.4f}".format(python_time))
print("Numpy version {:0.4f}".format(numpy_time))
print("Numpy is in this example {:0.4f} times faster!".format(python_time/numpy_time))


# In[2]:


import numpy as np
print(np)


# In[3]:


my_list = [1,2,3,4,5]                         #This is an example of python list
my_list


# In[4]:


type(my_list)                               # Type function is used to know the type of Python objects


# In[5]:


arr = np.array(my_list)                       # This is a one dimensional array  
arr


# In[6]:


type(arr)


# In[7]:


[1,2],[2,3]
[[1,0,5],[0,2,3]]
[[[0.5,0.5],[0.5,0.5,0.5,0.5]],[[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]]]


# In[8]:


my_mat = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]                           # Creating a 2D list or list of lists
my_mat


# In[9]:


mat = np.array(my_mat)                                       # This is a 2 dimensional array
mat


# In[10]:


type(mat)


# In[11]:


arr


# In[12]:


arr.shape


# In[13]:


mat


# In[14]:


mat.shape                                                    # This gives the number of rows and columns of the array.


# #### shape function is used to know the resolution of an array

# In[15]:


arr.ndim                                          #This is 1 dimension array


# In[16]:


mat.ndim                                         #This is 2 dimension array or matrix


# #### ndim function is for checking dimension

# In[17]:


arr.dtype                                   # The data type of elements inside array is int32


# In[18]:


mat.dtype


# ### 1.5. Array Initialization

# Generate a numpy array of 4 zeros

# Here we will learn generating arrays with varying **step size**.

# In[19]:


import numpy as np
ap = np.arange(0,10)
print(ap)


# Generate a numpy array with numbers between 0 and 10, as no step size is defined so it will take 1 as default step size.

# In[20]:


np.arange(0,10,2)


# As you can see a numpy array with numbers between 0 and 10 and a step size of 2 is generated.
# **NOTE:** There are 3 parameters within arange (start, stop, step) start and stop specifies range of the array, while step defines the distance between two consequetive values.

# In[21]:


np.zeros[3]


# In[23]:


np.zeros((3,3)).ndim 


# In[24]:


np.ones((3,3))


# In[25]:


np.linspace(0,5,10)


# In[26]:


np.eye(3)


# ### 1.6. Array Initialization using Random Numbers
# 
# Here we are going to see how to populate array with random numbers.
# Generate an array of 5 random numbers

# In[27]:


import numpy as np


# In[28]:


np.random.rand(5)  


# In[29]:


np.random.rand(2,2)


# In[30]:


np.random.randn(5,5)


# In[31]:


nine_random_values = np.random.rand(9)

print(nine_random_values)


# In[32]:


nine_random_values.reshape(9)


# ### 1.7. Numpy Indexing
# 
# Once we have learned how to create arrays in Numpy, lets see how they are indexed and how we can access the contained elements.

# In[33]:


my_arr = np.arange(0,10)
my_arr


# In[34]:


my_arr = np.arange(0,10)
print(my_arr)


# In[35]:


my_arr[1]                                               # It will return element at index 10


# In[36]:


my_arr


# In[37]:


my_arr[1:5]


# In[38]:


my_arr[7:]


# In[39]:


my_arr[:6]


# In[40]:


my_arr[0:5] = -5
my_arr


# In[41]:


arr_2d = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])  


# In[42]:


arr_2d        # arange 0-12 and then reshape 4,3


# In[43]:


arr_2d[0,2]


# In[44]:


arr_2d[0:2,0:2]   


# In[45]:


arr_2d[:2,1:]


# ### 1.7.2. Conditional Indexing in an Array
# Here we can filter elements of an array based on some conditions. Below are the steps:
# 
#  - First we need to create a boolean array based on an conditional statement using conditional operators for comparison.
#  - Then this boolean array is passed as index of the original array to return the filtered elements.

# In[46]:


import numpy as np


# In[47]:


arr_ = [1,2,3,4,5]
print(arr_)

'''
arr_>2
'''


# In[48]:


import numpy as np


# In[68]:


arr_cond = np.arange(2,10)


# In[50]:


arr_cond


# In[51]:


arr_cond>2


# In[52]:


arr_cond[[False,  True,  True,  True,  True,  True,  True,  True]]


# In[53]:


arr_cond>6


# In[54]:


arr_cond[arr_cond>6]


# In[55]:


len(arr_cond)


# In[56]:


arr_2d = np.array([[1,2,3],[5,6,7],[8,9,10],[12,13,14]])    # Creating numpy array
arr_2d


# In[57]:


arr_2d.shape


# In[58]:


scaler = 3                                      #scaler


# In[59]:


scaler


# In[60]:


arr_2d + 3


# In[61]:


arr_1d = np.array([[10,10,10]])                               # array with different shape
arr_1d.shape


# In[62]:


arr_1d = np.array([10,10,10])                               # array with different shape
arr_1d.shape


# In[63]:


print(arr_1d)
print()
print()
print(arr_2d)


# In[64]:


arr_2d + arr_1d


# arr_2d --> 4 * 3
# 
# arr_1d --> 1 * 3
# In this case the array with lower dimension i.e. arr_1d is stretched such that it matches the dimensions of arr_2d
# And then addition is performed. Lets learn it better with one diagramatic example.

# In[65]:


# arr --> 1 * 4
arr = np.array([[1,1,1,1,1,1,1,1]])
arr = arr.reshape(2,4)


# In[66]:


print(arr)
print()
print()
print(arr_2d)


# In[84]:


arr_2d + arr_1d                                              # operation with a array of different shape


# ## 1.8.2. Numpy Mathematical Functions
# 
# Here you can see a lot of commonly used built-in functions of numpy for mathematical operations. These functions are faster and optimized for large size arrays.

# In[70]:


import numpy as np
arr = np.arange(1,11)                          # Lets first create a numpy array
arr


# In[71]:


arr[0]=2


# In[72]:


arr


# In[73]:


arr.argmin()


# In[74]:


arr.argmax()                                     # Maximum of the array


# In[75]:


a = [5,2,3,4,1]
#a.min()
min(a)
#max(a)


# In[76]:


np.amin(arr)


# **Min and Max** are used to calculate the **range of array**

# In[77]:


arr.argmin()                         # Index position of minimum of array


# In[78]:


arr.argmax()                           # Index position of maximum of array


# **Argmax** can be used to observe the output of a **softmax layer in Neural Networks**

# In[79]:


np.sqrt(arr)


# **sqrt** is used to calculate **Root Mean Squared Error**

# In[80]:


arr.mean()


# In[81]:


np.exp(arr)                                         # To calculate exponential value of each element in an array


# **Reshape Function**

# In[82]:


arr = np.arange(0,16)
arr


# In[83]:


arr_2d = arr.reshape(4,2)
arr_2d


# In[85]:


arr_2d = arr.reshape(4,4)
arr_2d


# In[86]:


arr_2d.flatten()


# In[87]:


arr_2d.transpose()


# ### 1.3.8.2. Merging and Splitting Arrays

# In[88]:


arr_x = np.array([[1,2,3,4],[5,6,7,8]])                                       # Lets create 2 arrays
arr_y = np.array([[21,22,23,24],[25,26,27,28]])


# In[89]:


arr_x


# In[90]:


arr_y


# **Concatenate** is used to join 2 arrays either along rows or columns

# In[91]:


np.concatenate((arr_x, arr_y), axis=1)                 # Join 2 arrays along columns


# In[92]:


arr_z = np.concatenate((arr_x, arr_y), axis=0)         # Join 2 arrays along rows
arr_z


# In[93]:


print(arr_x)
print()
print(arr_y)
print()
print(arr_z)


# In[94]:


np.hsplit(arr_z, 2)                                   # It will split the array into 2 equal halves along the columns


# In[95]:


np.vsplit(arr_z, 2)                                   # It will split the array into 2 equal halves along the rows


# __Takeaways__<br>
# Using _concatenate_ function we can _merge_ arrays columnwise and rowwise. Also arrays can be horizontally and vertically spliited using _hsplit_ and _vsplit_.

# In[ ]:





# __Conclusion__<br>
# Numpy is open source add on module to Python.<br>
# - By using NumPy you can __speed up__ your workflow and _interface with other packages_ in the Python ecosystem that use NumPy under the hood.
# - A growing plethora of scientific and mathematical Python-based packages are using NumPy arrays; though these typically support   Python-sequence input, they convert such input to NumPy arrays prior to processing, and they often output NumPy arrays.
# - It provide common __mathematical and numerical routines__ in pre-compiled, fast functions. 
# - It provides _basic routines_ for manipulating __large arrays and matrices__ of numeric data.

# In[ ]:


__Key Features__<br/>
- NumPy arrays have a __fixed size__ decided at the time of creation. _Changing the size of an ndarray will create a new array and delete the original._
- The elements in a NumPy array are all required to be of the __same data type__, and thus will be the same size in memory.
- NumPy arrays facilitate __advanced mathematical__ and other types of __operations__ on large numbers of data.

