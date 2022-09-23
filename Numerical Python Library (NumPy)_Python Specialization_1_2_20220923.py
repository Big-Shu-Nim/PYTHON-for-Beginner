#!/usr/bin/env python
# coding: utf-8

# Numpy is the fundamental pacakge for numeric computing with Python. It provides powerful ways to create, store, and/or manupulate date, which makes it able to seamlessly and speedily intergrate with a wide variety of databases. This is also the foundation that Pandas is built on which is a high performance data centric package that we're going to learn more about in this course. 
# 
# In this lecture, we will talk about creating array with certain data type, manipulating array, selecting elements from arrays, and loading dataset into array. Such functions are useful for manupulating data and understanding the functionalities of other common Python data package. 
# 
# 

# In[1]:


# you'll recall that we import a library using the 'import' keyword as numpy's common abbreviation is np

import numpy as np
import math


# # Array Creation

# In[3]:


# Arrays are displayed as a list or list of lists and can be created through list as well. When creating an
# array, we pass in a list as an argument in numpy array
a = np. array([1,2,3])
print(a)
# we can print the number of dimentions of a list using the ndim attribute
print(a.ndim)


# In[5]:


# if we pass in a list of list in numpy array, we create a multi-dimensional array, for instance, a matrix
b = np. array([[1,2,3],[4,5,6]])
b


# In[6]:


# we can print out the length of each dismension by calling the shape attribute. which returns a tuple.
b.shape


# In[7]:


#We can also check the type of itmes in the array
a.dtype


# In[9]:


# Besides integers, floats are also accpeted in numpy arrays
c = np. array([2.2, 5, 1.1]) 
# we can put in some floting point numbers 
c.dtype.name


# In[10]:


# Let's look at the data in our array
c


# * Note that numpy automatically converts integers, like 5, up to floats, since there is no loss of precision. 
# 
# * Numpy will try and give you the best data type possible to keep your data type homogenous, which means all the same, in the array

# * Sometimes we know the shape of an array that we want to create, but not what we want to be in it. 
# * Numpy offers several functions to create arrays with initial placeholders, such as zeor's or one's.
# 

# In[12]:


# Lets create two arrays, both the same shape but with different filter values.

d = np.zeros((2,3))
print(d)

e = np.ones((2,3))
print(e)


# In[13]:


# you'll see zeros, ones and rand used quite often to create example arrays, especailly in tack overflow posts 
#and other forums.


# we can also create a sequence of numbers in an array with
# the arrange function. 
# 
# The first argument is strating bound and the second argument is the ending bound. and the third argument is the difference between 
# each consecutive numbers. 
# 

# In[18]:


#Let's create an array of every even number from 10 inclusive to 50, exclusive.
f = np.arange(19, 50, 2)
f


# * If we want to genenrate a seequence of floats, we use something called linspace
# * in this function, the third argument isn't the difference between two numbers, but it's the total number of items that you want to generate. 

# In[20]:


# So 
np.linspace(0, 2, 15) # 15 numbers from 0 (inclusive) to 2(inclusive)


# # Array Operations
# 

# In[21]:


# we can do many things on arrays, such as mathematical manipulation, addition, subtraction, square, exponents, as well as Boolean arrays
# we can also do matrix manipulations, suchas product transpose, inverse and so forth


# In[22]:


# Arithemetic operations on array apply elementwise.
# let's create a couple of arrays


# In[23]:


a = np.array([10,20,30,40])
b = np.array([1,2,3,4])

# now let's look at a minus b
c = a - b 
print(c)

# and let's look at a times  b
d = a*b
print(d)


# In[24]:


# with arithmetic manipulation, we can convert current data to the way we want it to be.
# problems I face _ I moved down to the United States about six years ago from Canada. In Canada, we use Celsius for temperatures
# and my wife still hasn't converted to US system, which uses Fahrenheit.
# with numpy I could easily convert a number of Fahrenheit values, say,
# The weather forecast to Celisius for her

#let's create an array of typical Ann Arbor winter Fahrenheit values.
fahrenheit = np.array([0, -10, -5, 15])

# the formula for conversion is the temperature ((F-32) =5/9 = C)
celcius = (fahrenheit - 31) *(5/9)
celcius


# In[25]:


# Great, we now she knows it's a little chily outisde this week


# In[27]:


#Another useful and important manipulation is the Boolean array.
# We can apply an operator on an array and a Boolean array will be returned for any element in the original with true
# being emitted if it meets the condtion. 
celcius > -20


# In[28]:


# here's another example, we could use the modulus operator to check 
# numbers in array to see if they're even, so celsius mod 2 equls 0.
celcius %2 == 0


# In[31]:


# Beside elementwise manupulations. it is important to know that numpy supports matrix manipulation.
# lelt's look at the matrix product, if we wanted to do elementwise product, we use the "*" sign.\
A = np.array([[1,1],[0,1]])
B = np.array([[2,0],[3,4]])
print(A*B)


# In[32]:


# if we want to do matrix product, we're going to use the @ sign instead of asterisk. 
# so the asterisks is for elementwise and this is really important.

print(A@B)


# In[33]:


# you dont have to worry about complex matrix operations for this course.
# but it's important to know that numpy is the underpinning of
#scientific computing libraries and python.
# element wise operations ( the asterisks) as well as matrix-level operations (the @ sign)
# There's more on this in a subsequent course. 


# In[36]:


#So a few more linear algebra concepts are woth layering in here.
# you might recall that the product of two matrices is only plausible
# when the inner dimensions of the two matrices are the same.
# The dimensions refer to the number of elements, both horizontal and vertical in the rendered matrices
# that you've been seeing here. 
# so we can use numpy to quickly see the shape of a matrix. 
A.shape


# In[41]:


# when manipulating arrays of different types, the tpye of the resulting array
# will correspond to the more general of the two types. This is called upcasting.

#Let's create an array of integers

array1= np.array([[1,2,3], [4,5,6]])
print(array1.dtype)

# now let's create an array of floats.

array2= np. array([[7.1, 8.2, 9.1], [10.4,11.2, 12.3]])
print(array2.dtype)


# * Integers (int) are whole numbers only, and Floating point numbers (float) can have a whole number portion
#   and a decimal portion. The 64 in this example refers to the humber of bits that the operating system is reserving
#   to represent the number, which determines the size(or precision) of the numbers that can be represented.

# In[42]:


# Let's do an addition for the two arrays
array3= array1+array2
print(array3)
print(array3.dtype)


# In[43]:


#Notice how the items in the resulting array have been upcast into floating point numbers. 


# In[44]:


#Numpy have many interesting aggregation functions on them, such as sum(), max(), min(), and mean()

print(array3.sum())
print(array3.max())
print(array3.min())
print(array3.mean())


# In[46]:


# For two dimensional arrays, we can do the same thing for each row or column
# let's create an array with 15 elements, ranging from 1 to 15,
# with a dimension of 3x5
b = np.arange(1,16,1).reshape(3,5)
print(b)


# In[47]:


# now we often think about two dimensional arrays being made up of rows and columns, but you can also think
# of these arrays as just a giant ordered list of numbers, and the *shape* of the array, the number of rows
# and columns, is just an abstraction that we have for a particular purpose. Actually, this is exactly how basic
# images are stored in computer enviornments. 


# In[49]:


# for this demonstration I'll use the python imaging library (PIL) and a fuction to display images in the 
  # Jupyter notebook

from PIL import Image
from IPython.display import display

# let's just look at the image I'm talking about
im = Image.open('test.jpeg')
display(im)


# In[50]:


# now we can conver this PIL image to a numpy array
array = np. array(im)
print(array.shape)
array


# In[51]:


#Here we see that we have 200*200 array and that the values are all unit8. The unit menas that they are 
#unsigned integers( so no negative numbers) and the 8 means 8 bits per byte. This means that each value can
# be up to 2*2*2*2*2*2*2*2 = 256 in size(well acutally 255), becasue we start at zero). FOr Black and White 
# images black is stored as 0 and white is stored as 255. So if we just wanted to invert this iamge we could
# use the numpy array to do so. 

mask= np.full(array.shape, 255)
mask


# In[55]:


# Now let's subtract that from the modifired array
modified_array=array-mask

#and lets convert all of the negative values to positive values 
modified_array=modified_array*-1

#and as a last step, let's tell numpy to set the value of the datatype corretly

modified_array=modified_array.astype(np.uint8)

modified_array


# In[56]:


# and lastly, lets display this new array, we do this by using the fromarray() function in the python
# imaging library to convert the numpy array into an object jupyer can render
display(Image.fromarray(modified_array))


# In[60]:


# Ok, remember how I started this by talking about how we could just think of this as a giant array
# of bytes, and that the shape was an abstraction/ well, we could just decide to reshape the array and still
# try and render it. PIL is interpreting the individual rows as lines, so we can change the umber of lines
# columns if we want to. What do  you think that would look like?

reshaped=np. reshape(modified_array,(952, 326))
print(reshaped.shape)
display(Image.fromarray(reshaped))


# In[61]:


# Cant say i find that particularly flattering. By reshaping the array to be only 100 rows high but 400
# columns we've essentially doubled the image by taking every other line and stacking them out in width. THis
# makes the image look more stretched out too.

# This isn't image manipulation course, but the point was to show you that these numpy arrays are really just
# abstractions on top of data, and that dat ahs an underlying format(in this case, uint8) but further,
# we can build abstractions on top of that, such as computer code which renders a byte as either black or
# white, which ahs meaning to people. in some ways, this whole degree is about data and the abstactions that 
# we can build on top of that data, from indivual byte represenations thorugh to comples neural networks of 
# functions or interactive visualziations. your role as a data scientist is to understand what the data means
# (it's context an collection) and transform it into a different represenation to be used for sensemaking.


# In[63]:


# ok back to the mechanics of numpy.


# # Indexing, Slicing and Iterating

# In[64]:


# Indexing, slicing and interating are extremely important for data manupulation and analysis becasue these 
# techiniques allow us to select data based on condtions, and copy or update data. 


# ## Indexing

# In[67]:


# First we are going to look at integer indexing. A one-dimensional array, works in similar ways as list
# To get an element in a one-dimensional array, we simply use the offset index.
a = np.array([1,3,5,7])
a[2]

# python indexing start at number zero


# In[68]:


# For multidimensional array, we need to use integer array indexing, let's create a new multidimensional array
a = np.array([[1,2],[3,4],[5,6]])
a


# In[74]:


# if we want to select one certain element, we can do so by enetering the index, which is comprised of two
# integers the first being the row, and the second the column
a[1,1] # remember in python we start at 0 !


# In[79]:


# if we want to get multiple elements
# for example, 1, 4, and 6 and put them into a one-dimensional array
# we can enter the indices directly into an array fucntion 
np.array([a[0,0], a[1,1], a[2,1]])


# In[80]:


# we can aloso do that by using another form of array indexing, which essentiall "zips" the first list and the 
# second list up 
print(a[[0, 1, 2], [[0, 1, 1]]])


# # Boolean Indexing

# In[81]:


# Boolean indexing allows us to select arbitrary elements based on conditions. For example, in the matrix we
# just talked about we want to find elements that are greater than 5 so we set up a condidtion a > 5 

print(a>5)

# This returns a boolean array showing that if the value at the corresponding index is greater than 5


# In[82]:


# we can then place this array of booleans like a mask over the original array to return a one-dimensional
# array relating to the tru values. 
print(a[a>5])


# In[83]:


# As we will see, this functionality is essential in the pandas toolkit which is the bulk of this course 


# # Slicing

# In[85]:


# slicing is a way to create a sub-array based on the original array. For one-dimensional arrays, slicing
# works in similar ways to a list. To slice, we use the : sign/ For instance, if we put :3 in the indexing
# brackets, we get elements from inex 0 to index 3 (excluding index 3)


# In[86]:


a = np.array([0,1,2,3,4,5])
print(a[:3])


# In[87]:


# By putting 2:4 in the bracket, we get elements from index 2 to index 4 (excluding index 4)
print(a[2:4])


# In[88]:


# For multi-dimensional arrays, it works similarly, lets see an example
a = np.array([[1,2,3,4,], [5,6,7,8], [9,10,11,12]])
a


# In[89]:


# First, if we put one arguemnt in the array, for example a[:2] then we would get all the elements from the first
# (0th) and second row(1th)
a[:2]


# In[90]:


# if we add another argument to the array, for example a[:2. 1:3], we get the first two rows but then the
# second and third column values only
a[:2, 1:3]


# In[91]:


#So, in multidimensional arrays, the first argument is for selecting rows, and the second argument is for 
# selecting columns 


# In[92]:


# It is important to realize that a slice of an array is a view into the same data. This is called passing by
# reference. So modifying the sub array will consequently modify the orifinal array. 

# Here I'll change the element at position [0, 0], which is 2, to 50, then we can see that the value in the 
# original array is changed to 50 as well

sub_array = a[:2, 1:2]
print("sub array index [0,0] value before change:", sub_array[0,0])

sub_array[0,0] = 50

print("sub array index [0,0] value after change:", sub_array[0,0])
print("original array index [0,1] value after change:", a[0,1])


# # Trying Numpy with Datasets

# In[93]:


# now that we have learned the essentials of Numpy let's use it on a couple of datasets.


# In[94]:


# Here we have a very popuular dataset on wine quality, and we are going to only look at red wines. The data
# fileds include: fixed acidity, volatile aciditycitric acid, residual sugar, cholorides, free sulfur dioxide, 
# total sulfur dixidedensity, ph, sulphates, alchol, quality 


# In[98]:


# To load a dataseet in Numpy, we can use the genfromtxt() fucntion. We can specify data file name, delimiter
# (which is optional but often used), and number of rows to skip if we have a header row, henc it is 1 here

# The genfromtxt() function has a parameter called dtype for specifying data types of each column this
# parameter is optional. Without specifying the types, all types will be casted the same to the more 
# general/precise type. 

wines = np. genfromtxt("resources/week-1/datasets/winequality-red.csv", delimiter=";", skip_header=1)
wines


# In[106]:


# Recall that we can use integer indexing to get a certain column or a row. For example, if we want to select
# the fixed acidity column, which is the first column, we can do so by entering the index into the array.
# Also remember that for multidimensional arrays, the first argument refers to the row, and the second
# argument refers to the colun, and if we just ive one argument then we'll get a single dimensional list back

# so all reows combied but only the first column from them would be
print("one interger 0 for slicing: " , wines[:, 0])
# but if we wanted the same valeus but wnated to preserve that they sit in their own rows we would write
print("0 to 1 for slicing: \n", wines[:, 0:1])


# In[101]:


# This is another great example of how the shape of the data is an abstraction which we can layer
# intentionally on top of the data we are working with.


# In[102]:


# If we want a range of columns in order, say columns 0 thorugh 3 (recall, this means first, second and)
# thrid, since we start at zero and don't include the traning index value), we  can do that too
wines[:, 0:3]


# In[103]:


# what if we want serveral non-consecutive columns? We can place the indices of the columns that we want into
# an array and pass the array as the second argument. Here's an example 
wines[:, [0,2,4]]


# In[107]:


# we can also do some basic summaraization of this dataset. For example, if we want to find out the average
# quality of red wine, we can select the quality column. we could do this in a couple of ways, but the most
# appropriate is to use the -1 value for the index, as negative numbers mean slicing from the back the list. 
# we can then call the aggregation fucntins on this data.
wines[:,-1].mean()


# In[108]:


import pandas as pd
df =  pd.read_csv("resources/week-1/datasets/winequality-red.csv")


# In[109]:


df.head(3)


# In[110]:


# Let's take a look at another dataset, this time on graduate school admissions. It has fields such as GRE
# score, TOEFL socre, university rating, GPA, having research experience or not, and a chance of admission.
# withs this dataset, we can do data manipulation and basic analysis to infer what condtions are associated
# with higher chance of admiision. Let's take a look.


# In[121]:


# we can specify data filed names when using genfromtxt() to loads CSV data. Also, we can have numpy try and
# infer the type of a column by setting the dtype parameter to None
graduate_admission = np. genfromtxt("resources/week-1/datasets/Admission_Predict.csv", delimiter=",", skip_header=1,
                                   names=('Serial No'
                                          , 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
                                         'LOR', 'CGPA', 'Research', 'Chance of Admit'))
graduate_admission


# In[113]:


# Notice that the resulting array is actually a one - dimensional array with 400 tuples
graduate_admission.shape


# In[114]:


# We can retrieve a column from the array using the column's name for example, let's get the CGPA column and
# only the first five values. 
graduate_admission['CGPA'][0:5]


# In[140]:


# Since the GPA in the dataset range from 1 to 10, and in the US it's more common to use a scale of up to 4,
# a common task might be to convert the GPA by dividing by 10 and then multiplying by 4
graduate_admission['CGPA'] = graduate_admission['CGPA'] /10 *4
graduate_admission['CGPA'] [0:20] # let's get 20 values


# In[131]:


# Recall boolean masking. We can use this to find out how many students have had research experiece by
# creating a boolean mask and passing it to the array indexing operator
len(graduate_admission[graduate_admission['Research']==1])
# len can count the number of elements in a list 


# In[134]:


# Since we have the data field chance of admission, which ranges from 0 to 1, we can try to see if students
# with high chance of admission (>0.8) on average have higher GRE score than those with lower chance of
# admission (<0.4)

# so first we use boolean masking to pull out only those students we are interested in based on their chance
print(graduate_admission[graduate_admission['Chance_of_Admit']>0.8]['GRE_Score'].mean())
print(graduate_admission[graduate_admission['Chance_of_Admit']<0.4]['GRE_Score'].mean())
print(graduate_admission[graduate_admission['Chance_of_Admit']>0.4]['GRE_Score'].mean())


# In[137]:


# Take a moment to reflect here, do you understand what is happening in these calls?

# When we do the boolean masking we are left with an array with tuples in it still, and numpy holds underneath
# this a list of the columns we specified and their name and indexes
graduate_admission[graduate_admission['Chance_of_Admit'] > 0.8]


# In[141]:


# Let's also do this with GPA
print(graduate_admission[graduate_admission['Chance_of_Admit'] > 0.8]['CGPA'].mean())
print(graduate_admission[graduate_admission['Chance_of_Admit'] < 0.4]['CGPA'].mean())


# In[142]:


# I guess on could have expected this. The GPA and GRE for students who have a higher chance of being admitted, at 
# least based on our cursory look here, seems to be higher. 


# So that's a bit of a whirwing tour of numpy, the core scientific computing library in python. Now, you're going to see a lot more of this kind of discussion, as the library we'll be focusing on in this course is pandas, which is build on top of numpy. Don't worry if ti didn't all sink in the first time, we're going to dig in to most of these 
# topics again with pandas. However, it's useful to know that many of the functions and capabilities of numpy are available to you within pandas. 
# 
