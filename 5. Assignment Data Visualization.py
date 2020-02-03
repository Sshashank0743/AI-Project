#!/usr/bin/env python
# coding: utf-8

# # DATA VISUALIZATION

# In[22]:


import pandas as pd
pokemon = pd.read_csv("E:/Insaid Project/Numphy/New folder/Term-1-master/Data/Casestudy/Pokemon.csv", index_col=0)
pokemon.head()


# ### UNIVARIATE PLOTTING

# ### 1. Create a bar chart using "Type 1" and its frequency.

# In[23]:


import pandas as pd
def create():
    pokemon['Type 1'].value_counts().plot.bar()
    return None
create()


# ### 2. Create a line chart using "HP" and its frequency.

# In[24]:


import pandas as pd
def create():
    pokemon['HP'].value_counts().sort_index().plot.line()
    return None
create()


# ### 3. Create an area chart using "Total" and its frequency.

# In[25]:


import pandas as pd
def create():
    pokemon['Total'].value_counts().sort_index().plot.area()
    return None
create()  


# ### 4. Create a histogram using "Total" and its frequency.

# In[26]:


import pandas as pd
def create():
    pokemon['Total'].plot.hist()
    return None
create()


# ## BIVARIATE PLOTTING

# ### 5. Create a scatter plot using "Attack" and "Defense".

# In[27]:


import pandas as pd
def create():
    pokemon.plot.scatter(x='Attack', y='Defense')
    return None
create()


# ### 6. Create a hex plot using "Attack" and "Defense" with gridsize of 25.

# In[28]:


import pandas as pd
def create():
    pokemon.plot.hexbin(x='Attack', y='Defense', gridsize=25)
    return None
create()


# ### 7. Create a stacked chart on "Legendary" and "Generation" columns, for stacking on "Attack" and "Defense" values

# In[29]:


import pandas as pd
def create():
    pokemon_stats_legendary = pokemon.groupby(['Legendary', 'Generation']).mean()[['Attack', 'Defense']]
    pokemon_stats_legendary.plot.bar(stacked=False)
    return None
create()


# In[30]:


import pandas as pd
def create():
    pokemon_stats_legendary = pokemon.groupby(['Legendary', 'Generation']).mean()[['Attack', 'Defense']]
    pokemon_stats_legendary.plot.bar(stacked=True)
    return None
create()


# ### 8. Create a bivariate plot hist and line on "HP", "Attack", "Defense", "Sp.Atk", "Sp.Def", "Speed", grouped by "Generation".

# In[31]:


import pandas as pd
def create():
    pokemon_stats_by_generation = pokemon.groupby('Generation').mean()[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
    pokemon_stats_by_generation.plot.hist()
    return None
create()


# In[32]:


import pandas as pd
def create():
    pokemon_stats_by_generation = pokemon.groupby('Generation').mean()[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
    pokemon_stats_by_generation.plot.line()
    return None
create()


# ### 9. Create a pairplot dropping "Name" and using "Legendary" as hue and implement a palette of your choice(size=2).

# In[33]:


import seaborn as sns
def create():
    sns.pairplot(pokemon.drop("Name", axis=1), hue="Legendary",palette="viridis", size=2)
    return None
create()


# ## MULTIVARIATE PLOTTING

# ### 10. Create a multivariate scatter plot using "Attack" and "Defense" having hue as "Legendary" and appropriate markers as 'x' and 'o'.

# In[34]:


import seaborn as sns
def create():
    sns.lmplot(x='Attack', y='Defense', hue='Legendary', markers=['x', 'o'],fit_reg=True, data=pokemon)
    return None
create()


# In[35]:


import seaborn as sns
def create():
    sns.lmplot(x='Attack', y='Defense', hue='Legendary', markers=['x', 'o'],fit_reg=False, data=pokemon)
    return None
create()


# ### 11. Create a box plot using "Generation" and "Total" with hue as "Legendary".

# In[36]:


import seaborn as sns
def create():
    sns.boxplot(x="Generation", y="Total", hue='Legendary', data=pokemon)
    return None
create()


# ### 12. Create a heatmap using "HP", "Attack", "Sp. Atk", "Defense", "Sp. Def", "Speed" .

# In[37]:


import seaborn as sns
def create():
    sns.heatmap(pokemon.loc[:, ['HP', 'Attack', 'Sp. Atk', 'Defense', 'Sp. Def', 'Speed']].corr(),annot=True)
    return None
create()


# ### 13. Create  a parallel coordinates chart on "Attack", "Sp. Atk", "Defense", "Sp. Def" based on "Psychic" and "Fighting" skills.

# In[38]:


import pandas as pd
from pandas.plotting import parallel_coordinates
def create():
    p = (pokemon[(pokemon['Type 1'].isin(["Psychic", "Fighting"]))]
         .loc[:, ['Type 1', 'Attack', 'Sp. Atk', 'Defense', 'Sp. Def']]
    )
    parallel_coordinates(p, 'Type 1')
    return 
create()


# ### 14. Create a swarmplot using "Generation" and "Defense" marking "Legendary" as hue and with any palette of your choice.

# In[39]:


import seaborn as sns
def create():
    sns.swarmplot(x="Generation", y="Defense", hue="Legendary", palette="gnuplot", data=pokemon)
    return None
create()


# # THE END
