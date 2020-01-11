#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


import numpy as np
import pandas as pd

countries = ['India','France','England', 'USA']                                   # Creating two lists containing name of the country and its capital
capitals = ['New Delhi','Paris','London','Washington']                                      
arr = np.array(capitals)                                                         # Making an array with the list capitals
dicts = {'a':10, 'b':20, 'c':30, 'd':40, 'e':50, 'f':60, 'g':70, 'h':80}         # Defining a dictionary


# **Creating a series from a list**

# In[4]:


pd.Series(data = capitals)                                                 # Creating Pandas Series


# In[5]:


pd.Series(data = capitals, index = countries)                              # Modifying series by adding index


# **Creating a series from Numpy array**

# In[6]:


pd.Series(arr) 


# In[7]:


pd.Series(dicts)


# ## 2.2.1. Series Indexing
# 
# Here you will observe how the elements of the series are indexed and how we can access any element using it's index.

# In[8]:


my_series = pd.Series(dicts)


# In[9]:


my_series.shape


# In[10]:


my_series


# In[11]:


my_series[0]


# In[12]:


my_series[4]


# In[13]:


my_series[:2]


# In[14]:


my_series[3:7]


# In[15]:


my_series['a']


# In[16]:


my_series['d']


# In[17]:


my_series[['a','b','c','d']]


# In[18]:


my_series['a':'c']


# In[19]:


my_series


# In[20]:


my_series + my_series


# In[21]:


dicts1 = {1:10, 2:20, 3:30}   # Defining a dictionary
s1 = pd.Series(dicts1)
s1[2]


# ### 2.3. DataFrames
# 
# DataFrames are __2D data structures__ having data aligned in tabular format.
# 
# - Data is aligned in rows (also called index) and columns and can store __any datatypes__ like int, string, float, boolean.
# 
# - They are highly _flexible_ and offer a lot of mathematical functions.

# In[22]:


df = pd.DataFrame()                                         # Creating an Empty DataFrame
print(df)


# ### Creating a DataFrame from a List

# In[23]:


fruits = ['Apple','Banana','Coconut','Dates']
fruits_df = pd.DataFrame(fruits, columns=['Fruit'])         # columns is used to set the column name
fruits_df


# ### Creating a DataFrame from Nested Lists

# In[24]:


people = [['Rick',60, 'O+'], ['Morty', 10, 'O+'], ['Summer', 45,'A-'], ['Beth',18,'B+']]
people_df = pd.DataFrame(people, columns=['Name', 'Age', 'Blood Group'])
print(people_df)


# ### Creating a DataFrame from a Dictionary

# In[25]:


#import pandas as pd
people = {'Name':['Rick', 'Morty', 'Summer', 'Beth'], 'Age':[60,10,45,18], 'Blood Group':['O+','O+','A-','B+']}
people_df = pd.DataFrame(people)       
people_df


# __Takeaways__<br>
# From the above examples you got familiar with creating dataframes using various inputs like<br/>
# - __Lists__
# - __nested lists__
# - __dicitionary__<br/>
# Also note that pandas dataframes are __mutable__ and potentially __hetrogenous tabular data structure__.

# In[ ]:





# ### 2.3.1. Loading files into DataFrame
# 
# DataFrames can load data from many types of files like csv, json, excel sheets, text, etc. Lets learn how to do this one by one, first by using:

# - **Comma Separated Values or CSV**

# In[26]:


import pandas as pd


# In[27]:


#csv_df = pd.read_csv('C:/user-name/Desktop/supermarkets.csv')                               # read_csv is used to read csv file
csv_df = pd.read_csv('https://raw.githubusercontent.com/insaid2018/Term-1/master/Data/Casestudy/supermarkets.csv')                               # read_csv is used to read csv file
csv_df


# - **JSON**

# In[28]:


json_df = pd.read_json('https://raw.githubusercontent.com/insaid2018/Term-1/master/Data/Casestudy/supermarkets.json')                             # read_json is used to read json file
json_df


# - **Excel Sheets**<br>
# Here we use an additional parameter __sheet name__

# In[29]:


get_ipython().system('pip install xlrd')


# In[30]:


import pandas as pd
excel_df = pd.read_excel('https://github.com/insaid2018/Term-1/blob/master/Data/Casestudy/supermarkets.xlsx?raw=true', sheet_name=0)              # read_excel is used to read excel file
excel_df


# - **Data Structure separated by semi-colon ;**

# In[31]:


csv_df = pd.read_csv('https://raw.githubusercontent.com/insaid2018/Term-1/master/Data/Casestudy/supermarkets-semi-colons.txt', sep=';')            # sep is used to separate the dataset
csv_df   


# - **CSV file from the web**

# In[32]:


web_df = pd.read_csv('https://raw.githubusercontent.com/insaid2018/Term-1/master/Data/Casestudy/supermarkets.csv')            # write the url of the csv file within ''.
web_df


# _Takeaways_<br>
# We have seen how to load data from various types of files.
# - for __csv_ file use __read_csv,
# - for __json__ file use __read_json__
# - for excel files use read_excel__ funnctions.<br/>
# Also we made use of the __sep__ argument to __segment__ the datasheet.

# ### 2.3.2. Attributes of a DataFrame
# 
# After loading your DataFrame you may be interested to know the __columns, shape, datatypes__.<br/> 
# There are many functions in pandas to check all the different attributes of a DataFrame

# In[33]:


import pandas as pd
pd.read_csv('C:\\Users\\nasa\\att.csv',sep=';')


# **Checking the number of rows and columns**

# In[34]:


people_df


# In[35]:


people_df.shape


# **Checking the datatypes of elements in each column**

# In[36]:


print(people_df)


# In[37]:


print(people_df.dtypes)


# **Checking the column names, number of records, datatype of records**

# In[38]:


people_df.info()


# In[39]:


people_df.describe()


# In[40]:


people_df.count()   


# In[41]:


people_df.index


# In[42]:


people_df.columns


# In[43]:


people_df


# In[44]:


new_people_df = people_df.set_index('Name')                
new_people_df


# In[45]:


import numpy as np
a=np.arange(15).reshape(3,5)
print(a)


# In[46]:


a.sum(axis=1)


# In[ ]:





# - __Selecting a specific column__

# In[47]:


people_df


# In[48]:


people_df.Name


# In[49]:


people_df['Blood Group']


# In[50]:


people_df['Score'] = [10,9,8,7]
people_df


# In[51]:


people_df['city'] = ['Delhi','Kanpur','Agra','Lucknow']
people_df


# In[52]:


people_df.head(2)


# In[53]:


import pprint
pp = pprint.PrettyPrinter()
pp.pprint(people_df)


# In[54]:


get_ipython().system('pip install ipython')
import IPython


# In[55]:


from IPython.display import display

# Assuming that dataframes df1 and df2 are already defined:
display(people_df)


# In[56]:


people_df[['Score','Age']]


# In[57]:


people_df


# In[58]:


people_df['sum'] = people_df['Score'] + people_df['Age']
people_df['sum']


# - **Column Deletion using drop**<br>
# - It has 2 parameters axis and inplace
#     -  **axis=1** implies delete __columnwise__,
#     - __axis=0__ implies delete __rowwise__ <br>
#     - __inplace=True__ implies __Modify__ the df<br>

# In[59]:


people_df


# In[60]:


people_df.drop('sum', axis=1, inplace=False)          # Drop Sum and modify the dataframe


# In[61]:


people_df


# In[62]:


people_df.drop('sum', axis=1, inplace=True)             # Drop Sum and modify the dataframe


# In[63]:


people_df


# In[64]:


del people_df['Score']
print(people_df)


# In[65]:


new_people_df = people_df.set_index('Name')
new_people_df


# In[66]:


people_df.drop(['Blood Group'],1)


# In[67]:


people_df


# In[68]:


new_people_df


# In[69]:


new_people_df.drop('Summer', axis=0)                             # Row Deletion using Row Index


# In[70]:


new_people_df


# In[71]:


new_people_df.drop('Summer', axis=0, inplace=True)


# In[72]:


new_people_df


# In[ ]:





# ### 2.3.4. Indexing in DataFrame
# There are 2 types of indexing in Pandas<br>
# * Using numbers(record or column index) using **iloc**
# * Using names(record name or column name) using **loc**

# In[73]:


market_df = pd.read_csv('http://pythonhow.com/supermarkets.csv')
market_df


# In[74]:


market_df = market_df.set_index ('ID')


# In[75]:


market_df


# In[76]:


market_df.index


# In[77]:


market_df.iloc[0:4,1:3]


# In[78]:


market_df.iloc[:,0:5]


# In[79]:


market_df.iloc[1:,0:5]


# In[80]:


market_df.iloc[0:4,:]                                     # Return all the rows from 0-4


# In[81]:


market_df


# In[82]:


market_df.loc[3:5,"City"]                              # Returns ID 1-4 and column Address


# In[83]:


market_df.loc[1:4,"Name"]   


# In[84]:


market_df.loc[1:5,"Address":"Country"]                    # Returns ID 1-4 and columns Address to Country


# In[85]:


market_df.loc[1:5,"City":"Name"]                    # Returns ID 1-4 and columns City to Name


# In[86]:


market_df.loc[:,"State":]                                 # Return all the columns from State onwards and all the rows


# __Takeaways__<br>
# You can use these operations if you want to __separate features__ into _numerical and categorical columns_.
# - As we have seen use _iloc and loc_ for indexing using _numbers and names_ respectively.
# - You can also perform __train-test-split__ on the dataframe.

# In[ ]:





# ### 2.3.5. Merging, Concatenating and Appending
# 
# In the previous section we saw how to add rows or columns.<br>
# Here we will see how to merge two dataframes.

# In[87]:


df1 = pd.DataFrame({
    'id':[1,2,3,4,5],
    'name':['a','b','c','d','e'],
    'sub':['sub1','sub2','sub3','sub4','sub5']
})
df2 = pd.DataFrame({
    'id':[1,2,3,4,5],
    'name':['b','c','d','e','f'],
    'sub':['sub3','sub4','sub5','sub6','sub7']
})


# In[88]:


print(df1)
print('')
print(df2)


# In[89]:


print(df1)
print('\n')
print(df2)


# ### Concatenating 2 DataFrames
# This is used to join 2 dataframes along _rows or columns_

# In[90]:


pd.concat([df1, df2], axis=0)                              # Joining two DataFrame along the rows


# In[91]:


pd.concat([df1, df2], axis=1)                                  # Joining 2 DataFrame along the columns


# ### Merging 2 DataFrames
# 
# This is used to join two dataframes based on any **column** as **key**.

# In[92]:


import pandas as pd
df1 = pd.DataFrame({                                       # Lets create 2 dataframes
    'emp_id':[1,2,3,4,5],
    'emp_name':['a','b','c','d','e'],
    'sub':['sub1','sub2','sub3','sub4','sub5']
})
df2 = pd.DataFrame({
    'dept_id':[1,2,3,4,5],
    'dept_name':['b','c','d','e','f'],
    'sub':['sub3','sub4','sub5','sub6','sub7']
})

print(df1)
print('\n\n')
print(df2)


# In[93]:


pd.merge(left=df1, right=df2, on='sub')                    # Joining 2 DataFrame using 'sub' as key


# In[94]:


pd.merge(left=df1, right=df2, on='sub',how='left')   


# In[95]:


pd.merge(left=df1, right=df2, on='sub', how='outer')    # left, right, outer


# In[96]:


pd.merge(left=df1, right=df2, on='sub', how='inner', left_on=None, right_on=None,)    # left, right, outer


# In[97]:


pd.merge(left=df1, right=df2, on='sub', how='outer', left_on=None, right_on=None,)    # left, right, outer


# From the below pictorial depiction see how two dataframes are merged.<br/>
# ![do.png](attachment:do.png)

# In[98]:


Mylist1 = [(1,10),(2,20),(3,30),(4,40),(5,50)]
labels1 = ['ID','NUM']
df1_x = pd.DataFrame.from_records(Mylist1,columns= labels1)
df1_x
Mylist2 = [(1,'A'),(6,'B'),(3,'C'),(8,'D'),(10,'E')]
labels2 = ['ID','ALPHA']
df2_x = pd.DataFrame.from_records(Mylist2,columns= labels2)
print(df1_x)
print('\n')
print(df2_x)

df_onlyleft = pd.merge(left=df1_x,right=df2_x,on='ID',how='left',indicator= True).query('_merge=="left_only"').drop('_merge',1)
df_onlyleft


# In[99]:


df_left = df1_x
df_right = df2_x
df_onlyleft2 = df_left.query('ID not in @df_right.ID')
df_onlyleft2


# ### Appending a row on a DataFrame
# Append is used to add rows on a DataFrame.<br>

# In[100]:


df3 = pd.DataFrame({'id':[10,11],                             # Lets modify our row to a DataFrame
                    'name':['x','z'], 
                    'sub':['sub10','sub11']})
df3


# In[101]:


df1


# In[102]:


df4 = df1
df5 = df3

df4 = df4.append(df5)  
df4


# In[103]:


df4.reset_index(inplace=True)
#del df4['index']
df4


# In[104]:


df1


# In[105]:


df4 = df1
df5 = df3
df4.append(df5, ignore_index=False)


# In[106]:


df4 = df1
df5 = df3
df4.append(df5, ignore_index=False, verify_integrity=False)


# **These functions are useful when we want to create a DataFrame by combining 2 datasets**

# ### 2.3.6. Conditionals in DataFrame
# - This is used to perform **comparisions** on the records of DataFrame.<br>
# - The output of comparision is of **boolean** datatype.<br>
# - You can use this boolean to filter out records from the DataFrame.
# 

# In[107]:


market_df


# In[108]:


market_df['Employees'] >=15                                                       # This returns a Boolean Series


# In[109]:


market_df[~(market_df['Employees'] <=15)]                                            # This returns all the rows for which the condition is True


# In[110]:


market_df[~(market_df['Employees'] >=15)]                                            # This returns all the rows for which the condition is True


# In[111]:


(market_df['Employees'] >=15)


# In[112]:


(market_df['State'] != 'CA 94119')


# In[113]:


market_df[(market_df['Employees'] >=15) & (market_df['State'] != 'CA 94119')]


# ### 2.3.7. Multi Index DataFrame
# 
# Until now you have seen DataFrames with a single index. Lets see how we can use more than 1 index to gain better insights.

# In[114]:


import pandas as pd

company_df = {
    'Company':['Google','Google','Google','Microsoft','Microsoft','Microsoft'],
    'Year' : [2008,2009,2010,2008,2009,2010],
    'Revenue' : [11,15,16,9,12,14],
    'Employee' : [300,400,500,350,450,550] 
}

cmp_df = pd.DataFrame(company_df)
cmp_df.set_index(['Company','Year'],inplace=True)


# In[115]:


#Company=['Google','Google','Google','Microsoft','Microsoft','Microsoft']
#Year = [2008,2009,2010,2008,2009,2010]
#Revenue = [11,15,16,9,12,14]
#Employee = [300,400,500,350,450,550]

#list(zip(Revenue, Employee))                                                        # Zip will collect one value from each of its container(Revenue, Employee)

#list(zip(Company, Year))

#hier_index = list(zip(Company, Year))                                               # These pair values will be our 2 indices
#hier_index

#hier_index = pd.MultiIndex.from_tuples(hier_index)
#hier_index

#multi_index_df = pd.DataFrame(data = list(zip(Revenue, Employee)), index = hier_index, columns=['Revenue','Employee'])
#multi_index_df

#multi_index_df.index.names =['Company','Year']                                      # Rename our indices to Company and Year
#multi_index_df 

#multi_index_df.loc['Google']                                                        # Accessing data using our first index
#multi_index_df.loc['Google'].loc[2009]                                              # Accessing data using first index and then second index
#multi_index_df.loc[2009]                                              # Error Accessing data using first index and then second index

#multi_index_df.xs('Microsoft', level='Company')                                     # Accessing data based on Index level (either from first or second index)
#multi_index_df.xs(2008, level='Year')
#multi_index_df.xs(2009, level='Year')                                               # Accessing data based on Index level (either from first or second index)
#multi_index_df.xs(2010, level='Year')                                               # Accessing data based on Index level


# In[116]:


cmp_df.loc['Google']


# In[117]:


cmp_df.loc['Google'].loc[2009]


# In[118]:


cmp_df.xs('Microsoft', level='Company')                                     # Accessing data based on Index level (either from first or second index)


# In[119]:


cmp_df.xs(2009, level='Year')                                    # Accessing data based on Index level (either from first or second index)


# In[120]:


cmp_df.xs(2010, level='Year')                                               # Accessing data based on Index level


# In[121]:


cmp_df.xs(2008, level='Year')                                               # Accessing data based on Index level


# Lets observe Multi level DataFrame in another dataset

# In[122]:


import seaborn as sns                                                               # Import seaborn library for dataset
import pandas as pd
tips = sns.load_dataset('tips')
tips.head()                                                                         # head() shows the top 5 records of the dataframe


# In[123]:


tips.info()


# In[124]:


tips.describe()


# In[125]:


tips.time.unique()


# In[126]:


tips[(tips.time =='Lunch')]['tip'].sum()


# In[127]:


tips.shape


# In[128]:


tips.groupby(['time','sex','day'])['tip'].agg(['sum','mean','count'])


# In[129]:


tips.shape


# In[130]:


tips.index                                                                         # This dataset has a numeric index having range between 0 and 244


# In[131]:


tip = tips.head(5)                                                                # Lets create a smaller dataset from the first 5 rows
tip


# In[132]:


tip = tips.head(11)                                                                # Lets create a smaller dataset from the first 11 rows
tip


# In[133]:


tip.head()


# In[134]:


tip.set_index(['sex'])                                                             # Usually you can create an Index using a categorical variable


# In[135]:


multi_index_tips = tip.set_index(['sex','size'])                                   # Lets set 'sex' and 'size' as our index
multi_index_tips


# In[136]:


multi_index_tips.sort_index(inplace = True)                                        # Sorting all the index of serving size in ascending order
multi_index_tips


# In[137]:


multi_index_tips.loc['Male']                                                       # Accessing records if index is Male


# In[138]:


multi_index_tips.loc['Female']                                                       # Accessing records if index is Female


# In[139]:


multi_index_tips.xs(2, level='size')                                               # Accessing records if serving size is 2 (From second index)


# ### 2.3.8. Groupby
# 
# Groupby allows you to group together _rows_ based on a _column_ and perform an aggregate function on them.<br/>
# This is quiet a handy tool if you don't want to change the index of the DataFrame.

# In[140]:


tips.time.unique()


# In[141]:


tips.head()


# In[142]:


byTime = tips.groupby('time')                                                    # Lets analyse the metrics bases on time of meal(Lunch, Dinner)


# In[143]:


byTime


# In[144]:


byTime.count()                                                                   # Number of meals for Lunch and Dinner


# In[145]:


byTime.mean()                                                                    # Average bill


# In[146]:


tips.head(1)


# In[147]:


byTime.sum()                                                                     # Total bill


# In[148]:


tips.groupby('time').sum()


# Picture of _groupby_ function.
# 
# ![download%20%281%29.png](attachment:download%20%281%29.png)

# ### 2.3.9. Operations on a DataFrame
# Here are some basic operations you can perform on a DataFrame

# In[149]:


tips.head()                                                                     # Observe the first 5 elements of the dataframe


# In[150]:


tips['size'].unique()                                                           # Observe the unique values of tips['size']


# In[151]:


tips['day'].unique()


# In[152]:


tips['size'].nunique()                                                          # Observe the number of unique values of tips['size']


# In[153]:


tips['time'].value_counts()                                                     # Observe the number of counts of tips['size']


# ### Applying a function on DataFrame
# We can apply any function on the elements of a dataframe

# In[154]:


people = [['Rick',60, 'O+'], ['Morty', 10, 'O+'], ['Summer', 45,'A-'], ['Beth',18,'B+']]
people_df = pd.DataFrame(people, columns=['Name','Score', 'Blood Group'])
people_df


# In[155]:


people_df['Score']*2


# In[156]:


def times2(x):                                                                # We are going to apply this function
    num = x
    if x%6==0:
        num *= 2
    return num


# In[157]:


people_df['Score']


# In[158]:


people_df['Score'].apply(times2)                                              # Applying times2() function on a column of dataframe


# In[159]:


people_df['Score'].apply(lambda x: x * 2)                                     # Applying lambda function on a column of dataframe


# In[160]:


people_df.sort_values('Score')                                                # Sorting the records based on a column


# ### 2.4. Time Series in Pandas
# Here we will explore the __DateTime__ functions provided by Pandas and how efficient it is to analyse Time Series data.

# In[161]:


import pandas as pd
air_df = pd.read_csv('https://raw.githubusercontent.com/insaid2018/Term-1/master/Data/Casestudy/AirQualityUCI.csv')                                      # Import the Dataset


# In[162]:


air_df.head()


# In[163]:


type(air_df['CO(GT)'])


# In[164]:


air_df.describe()


# In[165]:


air_df[air_df['NMHC(GT)']==-200]['NMHC(GT)'].count()


# Almost 90% of the elements of NMHC(GT) is -200.0. Therefore dropping this column.

# In[166]:


air_df.drop(['NMHC(GT)'], axis=1)


# In[168]:


#air_df.head()                                                                  # Observe the columns and rows of the dataset


# In[169]:


air_df.info()


# In[170]:


air_df.head(1)


# In[171]:


air_df.count(1)


# In[175]:


air_df['Date'] = pd.to_datetime(air_df['Date'], format='%m/%d/%Y')     


# In[176]:


air_df.info()


# In[177]:


air_df['Time'].unique()     


# In[178]:


air_df.Time.unique()


# Converting __time__ to pandas __datetime__ with format HH/MM/SS

# In[179]:


air_df['Time'] = pd.to_datetime(air_df['Time'], format = '%H:%M:%S')     


# In[180]:


air_df.dtypes                                         # Final check for our date and time column


# In[181]:


year = air_df.Date.dt.year                            # Extracting Year from Date column
print(year.head())


# In[182]:


month = air_df.Date.dt.month                          # Extracting Month from Date column
print(month.head())


# In[183]:


month.nunique()                                       # Counting the number of months


# In[184]:


day = air_df.Date.dt.day                              # Extracting Day from Date column
print(day.head())    


# In[185]:


day.nunique()                                         # Counting the number of days


# In[186]:


week = air_df.Date.dt.week                            # Extracting week number from Date column
print(week.head())


# In[187]:


week.unique()


# In[188]:


day_of_week = air_df.Date.dt.dayofweek                 # Extracting the day of the week number
print(day_of_week.head())


# In[189]:


day_name = air_df.Date.dt.weekday_name                 # Extracting the name of the day
print(day_name.head())


# In[190]:


day_of_year = air_df.Date.dt.dayofyear                 # Extracting the day of the year
print(day_of_year.head())


# In[191]:


hour = air_df.Time.dt.hour                             # Extracting the hour from time
print(hour.head())


# In[192]:


hour.nunique()                                         # Counting the number of hours


# In[193]:


minute = air_df.Time.dt.minute                         # Extracting the minutes from the time
print(minute.head())


# In[194]:


second = air_df.Time.dt.second                         # Extracting the seconds from the time
print(second.head())


# Performing Conditional operations on Time. <br>
# Lets measure the number of records before 9 a.m.

# In[195]:


timestamp = pd.to_datetime("09:00:00", format='%H:%M:%S')   
air_df[air_df['Time'] < timestamp].shape


# Performing Conditional operations on Date. <br>
# Lets measure the number of records before 01/01/2005

# In[197]:


datestamp = pd.to_datetime("01/01/2005", format='%d/%m/%Y')


# In[198]:


from datetime import timedelta
#datestamp + timedelta(days=1)
#datestamp + timedelta(days=-1)


# In[199]:


air_df[air_df['Date'] < datestamp].tail()


# __Conclusion__<br/>
# Python is a great language for doing data analysis, primarily because of the fantastic ecosystem of data-centric Python packages.<br/>
# - __Pandas__ is one of those packages, and makes importing and analyzing data much easier.
# - It is an __open-source__, BSD-licensed Python library providing __high-performance__, __easy-to-use data structures__ and data analysis tools for the Python programming language. 
# - Python with Pandas is used in a wide range of fields including _academic and commercial domains_ like __finance, economics,         statistics, analytics__, etc.
# - They are built on packages like __NumPy and matplotlib__ to give you a single, convenient, place to do most of your data analysis and visualization work.

# __Key Features__<br/>
# - __Fast and efficient DataFrame object__ with default and customized indexing.
# - Tools for loading data into in-memory data objects from different file formats.
# - Data alignment and integrated handling of __missing data__.
# - __Reshaping and pivoting__ of date sets.
# - __Label-based slicing__, __indexing and subsetting__ of large data sets.
# - Columns from a data structure can be _deleted or inserted_.
# - __Groupby__ data for aggregation and transformations.
# - High performance __merging and joining__ of data.
# - __Time Series__ functionality.

# In[200]:


import pandas as pd
myList1 = [['M00000032+5737103','4264',    '0.000000',    '0.000000','N7789','10.905','10.635'],
    ['2M00000068+5710233',  '4264', '8.000000',    '-18.000000', 'N7789','10.664','10.132'],
    ['2M00000222+5625359',  '4264','0.000000',    '0.000000','N7789','11.982','11.433'],
    ['2M00000818+5634264',  '4264','0.000000',    '0.000000','N7789','12.501','11.892'],
    ['2M00001242+5524391',  '4264','0.000000',    '-4.000000','N7789','12.091','11.482']]
df1_st = pd.DataFrame(myList1,columns=['Star_ID','Loc_ID','pmRA','pmDE','Field','Jmag','Hmag']) 

print(df1_st)
myList2 = [['M00000032+5737103','4264',    '0.000000',    '0.000000','N7789','10.905','10.635'],
    ['2M00000068+5710233',  '4264', '8.000000',    '-18.000000', 'N7789','10.664','10.132'],
    ['2M00001242+5524391',  '4264','0.000000',    '-4.000000','N7789','12.091','11.482']]
df2_st = pd.DataFrame(myList2,columns=['Star_ID','Loc_ID','pmRA','pmDE','Field','Jmag','Hmag'])

print(df2_st)
df3_st = pd.merge(df1_st,df2_st,on='Star_ID')
print(df3_st)
df1_st = df1_st[~df1_st.Star_ID.isin(df3_st.Star_ID)]


# In[201]:


df1_st

