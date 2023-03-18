#!/usr/bin/env python
# coding: utf-8

# Visualization of different data types with python
# ==========
# Here, will learn some of the most basic `plotting` functionalities with `Python`, to give you the tools you need to assess basic distributions and relationships within you dataset. We will focus on the [Seaborn library](https://seaborn.pydata.org/index.html), which is designed to make nice looking `plots` quickly and (mostly) intuitively.

# In[1]:


import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# Let's first gather our dataset. We'll use participant related information from the [OpenNeuro dataset ds000228 "MRI data of 3-12 year old children and adults during viewing of a short animated film"](https://openneuro.org/datasets/ds000228/versions/1.0.0) .

# In[20]:


get_ipython().run_cell_magic('bash', '', 'curl https://openneuro.org/crn/datasets/ds000228/snapshots/1.0.0/files/participants.tsv  -o /data/participants.tsv\n')


# In[21]:


pheno_file = ('/data/participants.tsv')

pheno = pandas.read_csv(pheno_file,sep='\t')

pheno.head()


# What are our different variables?

# In[22]:


pheno.columns


# ### Univariate visualization

# Let's start by having a quick look at the `summary statistics` and `distribution` of `Age`:

# In[23]:


print(pheno['Age'].describe())


# In[24]:


# simple histogram with seaborn
sns.displot(pheno['Age'],
            #bins=30,          # increase "resolution"
            #color='red',    # change color
            #kde=False,        # get rid of KDE (y axis=N)
            #rug=True,         # add "rug"
            )


# What kind of distribution do we have here? 
# 
# Let's try log normalization as a solution. Here's one way to do that:

# In[25]:


import numpy as np

log_age = np.log(pheno['Age'])
sns.distplot(log_age,
            bins=30,          
            color='black',    
            kde=False,         
            rug=True,          
            )


# There is another approach for log-transforming that is perhaps better practice, and generalizable to *nearly any* type of transformation. With [sklearn](https://scikit-learn.org/stable/index.html), you can great a custom transformation object, which can be applied to different datasets.
# 
# _Advantages_ :
# * Can be easily reversed at any time
# * Perfect for basing transformation off one dataset and applying it to a different dataset
# 
# _Distadvantages_ :
# * Expects 2D data (but that's okay)
# * More lines of code :(

# In[26]:


from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, validate=True)

age2d = pheno['Age'].values.reshape(-1,1)
log_transformer.fit(age2d)

sk_log_Age = log_transformer.transform(age2d)


# Are two log transformed datasets are equal?

# In[27]:


all(sk_log_Age[:,0] == log_age)


# And we can easily reverse this normalization to return to the original values for age.

# In[28]:


reverted_age = log_transformer.inverse_transform(age2d)


# The inverse transform should be the same as our original values:

# In[29]:


all(reverted_age == age2d)


# Another strategy would be `categorization`. Two type of `categorization` have already been done for us in this dataset. We can visualize this with `pandas value_counts()` or with `seaborn countplot()`:

# In[30]:


# Value counts of AgeGroup
pheno['AgeGroup'].value_counts()


# In[31]:


# Countplot of Child_Adult

sns.countplot(pheno['Child_Adult'])


# ### Bivariate visualization: Linear x Linear

# Cool! Now let's play around a bit with `bivariate visualization`. 
# 
# For example, we could look at the association between `age` and a cognitive phenotype like `Theory of Mind` or `"intelligence"`. We can start with a `scatterplot`. A quick and easy `scatterplot` can be built with `regplot()`:

# In[32]:


sns.regplot(x=pheno['Age'], y=pheno['ToM Booklet-Matched'])


# `regplot()` will automatically drop missing values (`pairwise`). There are also a number of handy and very quick arguments to change the nature of the plot:

# In[33]:


## Try uncommenting these lines (one at a time) to see how
## the plot changes.

sns.regplot(x=pheno['Age'], y=pheno['ToM Booklet-Matched'],
           order=2,        # fit a quadratic curve
           #lowess=True,    # fit a lowess curve
           #fit_reg = False # no regression line
           #marker = ''     # no points
           #marker = 'x',   # xs instead of points
           )


# Take a minute to try plotting another set of variables. Don't forget -- you may have to change the data type!

# In[34]:


#sns.regplot(x=, y=)


# This would be as good a time as any to remind you that `seaborn` is built on top of `matplotlib`. Any `seaborn` object could be built from scratch from a `matplotlib` object. For example, `regplot()` is built on top of `plt.scatter`:

# In[35]:


plt.scatter(x=pheno['Age'], y=pheno['ToM Booklet-Matched'])


# If you want to get really funky/fancy, you can play around with `jointplot()` and change the `"kind"` argument.
# 
# However, note that `jointplot` is a different `type` of `object` and therefore follows different rules when it comes to editing. More on this later ...

# In[36]:


for kind in ['scatter','hex']: #kde
    sns.jointplot(x=pheno['Age'], y=pheno['ToM Booklet-Matched'],
                  kind=kind)

    plt.show()


# That last one was a bit weird, eh? These `hexplots` are really built for larger sample sizes. Just to showcase this, let's plot a `hexplot` 1000 samples of some `random data`. Observe how the `hexplot` deals with `density` in a way that the `scatterplot` cannot.

# In[37]:


mean, cov = [0, 1], [(1, .5), (.5, 1)]
x, y = np.random.multivariate_normal(mean, cov, 1000).T
sns.jointplot(x=x, y=y, kind="scatter")
sns.jointplot(x=x, y=y, kind="hex")



# More on dealing with "overplotting" here: https://python-graph-gallery.com/134-how-to-avoid-overplotting-with-python/.

# However, note that `jointplot` is a different type of object and therefore follows different rules when it comes to editing. This is perhaps one of the biggest drawbacks of `seaborn`.
# 
# For example, look at how the same change requires different syntax between `regplot` and `jointplot`:

# In[38]:


sns.regplot(x=pheno['Age'], y=pheno['ToM Booklet-Matched'])
plt.xlabel('Participant Age')


# In[39]:


g = sns.jointplot(x=pheno['Age'], y=pheno['ToM Booklet-Matched'],
                  kind='scatter')
g.ax_joint.set_xlabel('Participant Age')


# Finally, `lmplot()` is another nice `scatterplot` option for observing `multivariate interactions`.
# 
# However, `lmplot()` cannot simply take two `arrays` as input. Rather (much like `R`), you must pass `lmplot` some data (in the form of a `pandas DataFrame` for example) and `variable` names. Luckily for us, we already have our data in a `pandas DataFrame`, so this should be easy.
# 
# Let's look at how the relationship between `Age` and `Theory of Mind` varies by `Gender`. We can do this using the `"hue"`, `"col"` or `"row"` arguments: 

# In[40]:


sns.lmplot(x='Age', y = 'ToM Booklet-Matched', 
           data = pheno, hue='Gender')


# Unfortunately, these plots can be a bit sub-optimal at times. The `regplot` is perhaps more flexible. You can read more about this type of plotting here: https://seaborn.pydata.org/tutorial/distributions.html.

# ### Bivariate visualization: Linear x Categorical

# Let's take a quick look at how to look at `bivariate relationships` when one `variable` is `categorical` and the other is `scalar`.
# 
# For consistency can continue to look at the same relationship, but look at `"AgeGroup"` instead of `age`.
# 
# There are many ways to visualize such relationships. While there are some advantages and disadvantes of each type of plot, much of the choice will come down to personal preference.

# In[ ]:


sns.


# Here are several ways of visualizing the same relationship. Note that adults to not have cognitive tests, so we won't
# include adults in any of these plots. Note also that we explicitly pass the order of x:
# 

# In[42]:


order = sorted(pheno.AgeGroup.unique())[:-1]
order


# In[43]:


order = sorted(pheno.AgeGroup.unique())[:-1]

sns.barplot(x='AgeGroup', 
            y = 'ToM Booklet-Matched',
            data = pheno[pheno.AgeGroup!='Adult'])
plt.show()

sns.boxplot(x='AgeGroup', 
            y = 'ToM Booklet-Matched',
            data = pheno[pheno.AgeGroup!='Adult'])
plt.show()

sns.boxenplot(x='AgeGroup', 
            y = 'ToM Booklet-Matched',
            data = pheno[pheno.AgeGroup!='Adult'],
            order = order)
plt.show()

sns.violinplot(x='AgeGroup', 
            y = 'ToM Booklet-Matched',
            data = pheno[pheno.AgeGroup!='Adult'])
plt.show()

sns.stripplot(x='AgeGroup', jitter=True,
            y = 'ToM Booklet-Matched',
            data = pheno[pheno.AgeGroup!='Adult'])
plt.show()

sns.pointplot(x='AgeGroup', 
            y = 'ToM Booklet-Matched',
            data = pheno[pheno.AgeGroup!='Adult'])
plt.show()


# Generally, `lineplots` and `barplots` are frowned upon because they do not show the actual data, and therefore can mask troublesome distributions and outliers. 
# 
# But perhaps you're really into `barplots`? No problem! One nice thing about many `seaborn plots` is that they can be overlaid very easily. Just call two plots at once before doing `plt.show()` (or in this case, before running the cell). Just overlay a `stripplot` on top!

# In[44]:


sns.barplot(x='AgeGroup', 
            y = 'ToM Booklet-Matched',
            data = pheno[pheno.AgeGroup!='Adult'],
            order = order, palette='Blues')

sns.stripplot(x='AgeGroup', 
            y = 'ToM Booklet-Matched',
            data = pheno[pheno.AgeGroup!='Adult'],
            jitter=True,
            order = order, color = 'black')


# You can find more info on these types of plots here: https://seaborn.pydata.org/tutorial/categorical.html.
# 
# Having trouble deciding which type of plot you want to use? Checkout the raincloud plot, which combines multiple types of plots to achieve a highly empirical visualization. 
# 
# Read more about it here:
# https://wellcomeopenresearch.org/articles/4-63/v1?src=rss.

# In[48]:


get_ipython().run_cell_magic('bash', '', 'pip install ptitprince\n')


# In[49]:


import ptitprince as pt

dx = "AgeGroup"; dy = "ToM Booklet-Matched"; ort = "v"; pal = "Set2"; sigma = .2
f, ax = plt.subplots(figsize=(7, 5))

pt.RainCloud(x = dx, y = dy, data = pheno[pheno.AgeGroup!='Adult'], palette = pal, bw = sigma,
                 width_viol = .6, ax = ax, orient = ort)


# ### Bivariate visualization: Categorical x Categorical

# What if we want to observe the relationship between two `categorical variables`? Since we are usually just looking at `counts` or `percentages`, a simple `barplot` is fine in this case.
# 
# Let's look at `AgeGroup` x `Gender`. `Pandas.crosstab` helps sort the data in an intuitive way. 

# In[50]:


pandas.crosstab(index=pheno['AgeGroup'],
                columns=pheno['Gender'],)


# We can actually plot this directly from `pandas`.

# In[51]:


pandas.crosstab(index=pheno['AgeGroup'],
                columns=pheno['Gender'],).plot.bar()


# The above plot gives us absolute `counts`. Perhaps we'd rather visualize differences in `proportion` across `age groups`. Unfortunately we must do this manually.

# In[52]:


crosstab = pandas.crosstab(index=pheno['AgeGroup'],
                columns=pheno['Gender'],)

crosstab.apply(lambda r: r/r.sum(), axis=1).plot.bar()


# ### Style points

# You will be surprised to find out exactly how customizable your `python plots` are. Its not so important when you're first `exploring` your data, but `aesthetic value` can add a lot to `visualizations` you are communicating in the form of `manuscripts`, `posters` and `talks`.
# 
# Once you know the relationships you want to `plot`, spend time adjusting the `colors`, `layout`, and fine details of your `plot` to `maximize interpretability`, `transparency`, and if you can spare it, `beauty`!
# 
# You can easily edit `colors` using many `matplotlib` and `python arguments`, often listed as `col`, `color`, or `palette`. 

# In[53]:


## try uncommenting one of these lines at a time to see how the 
## graph changes

sns.boxplot(x='AgeGroup', 
            y = 'ToM Booklet-Matched',
            data = pheno[pheno.AgeGroup!='Adult'],
            #palette = 'Greens_d'
            #palette = 'spectral',
            #color = 'black'
           )


# You can find more about your palette choices here: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html.
# 
# More about your color choices here:
# https://matplotlib.org/3.1.0/gallery/color/named_colors.html.

# You can also easily change the style of the plots by setting `"style"` or `"context"`:

# In[54]:


sns.set_style('whitegrid')
sns.boxplot(x='AgeGroup', 
            y = 'ToM Booklet-Matched',
            data = pheno[pheno.AgeGroup!='Adult'],
           )


# In[55]:


sns.set_context('notebook',font_scale=2)
sns.boxplot(x='AgeGroup', 
            y = 'ToM Booklet-Matched',
            data = pheno[pheno.AgeGroup!='Adult'],
           )


# Notice these changes do not reset after the `plot` is shown. To learn more about controlling `figure aesthetics`, as well as how to produce temporary style changes, visit here: https://seaborn.pydata.org/tutorial/aesthetics.html.

# Finally, remember that these `plots` are `extremely customizable`. Literally every aspect can be changed. Once you know the relationship you want to `plot`, don't be afraid to spend a good chunk of time `tweaking` your `plot` to perfection:

# In[56]:


# set style
sns.set_style('white')
sns.set_context('notebook',font_scale=2)

# set figure size
plt.subplots(figsize=(7,5))

g = sns.boxplot(x='AgeGroup', 
            y = 'ToM Booklet-Matched',
            hue = 'Gender',
            data = pheno[pheno.AgeGroup!='Adult'],
           palette = 'viridis')

# Change X axis
new_xtics = ['Age 4','Age 3','Age 5', 'Age 7', 'Age 8-12']
g.set_xticklabels(new_xtics, rotation=90)
g.set_xlabel('Age')

# Change Y axis
g.set_ylabel('Theory of Mind')
g.set_yticks([0,.2,.4,.6,.8,1,1.2])
g.set_ylim(0,1.2)

# Title
g.set_title('Age vs Theory of Mind')

# Add some text
g.text(2.5,0.2,'F = large #')
g.text(2.5,0.05,'p = small #')

# Add significance bars and asterisks
plt.plot([0,0, 4, 4], 
         [1.1, 1.1, 1.1, 1.1], 
         linewidth=2, color='k')
plt.text(2,1.08,'*')

# Move figure legend outside of plot

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# That's all for now. There's so much more to visualization, but this should at least get you started.

# #### Recommended reading:
# 
# multidimensional plotting with seaborn: https://jovianlin.io/data-visualization-seaborn-part-3/
# 
# Great resource for complicated plots, creative ideas, and data!: https://python-graph-gallery.com/
# 
# A few don'ts of plotting: https://www.data-to-viz.com/caveats.html
