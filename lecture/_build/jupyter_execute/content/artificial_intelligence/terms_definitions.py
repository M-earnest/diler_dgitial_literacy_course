#!/usr/bin/env python
# coding: utf-8

# # Models, AI and all other buzz words
# 
# 

# What do these faces have in common?
# 
# <center><img src="https://static01.nyt.com/images/2020/11/19/us/artificial-intelligence-fake-people-faces-promo-1605818328743/artificial-intelligence-fake-people-faces-promo-1605818328743-superJumbo-v2.jpg" alt="logo" title="Github" width="400" height="280" /></br>
# <sub><sup><sub><sup><sup>https://static01.nyt.com/images/2020/11/19/us/artificial-intelligence-fake-people-faces-promo-1605818328743/artificial-intelligence-fake-people-faces-promo-1605818328743-superJumbo-v2.jpg
# </sup></sup></sub></sup></sub></center>
# 
# 

# What is `AI`?
# 
# <center><img src="https://external-preview.redd.it/hWh_8TpqrT6zAwpzHJ_m9Rx3iHjc_yI4zSI6aazMFTc.jpg?auto=webp&s=6d8006ac3edca5bad98dd7b5b9a4a8d5554eaff0" alt="logo" title="Github" width="400" height="280" /></br>
# <sub><sup><sub><sup><sup>https://external-preview.redd.it/hWh_8TpqrT6zAwpzHJ_m9Rx3iHjc_yI4zSI6aazMFTc.jpg?auto=webp&s=6d8006ac3edca5bad98dd7b5b9a4a8d5554eaff0
# </sup></sup></sub></sup></sub></center>
# 
# 
# 
# 

# Quite often the discussion about `AI` centers around `general artificial intelligence` which might be a bit of an ill-posed problem as we don't have any reason that "true" `general artificial intelligence` can exist. The focus here is on `"general"`. 
# 
# <img align="center" src="https://cdn-images-1.medium.com/fit/t/1600/480/1*rdVotAISnffB6aTzeiETHQ.png" alt="logo" title="Github" width="900" height="300" />
# 
# <center><sub><sup><sub><sup><sup>https://cdn-images-1.medium.com/fit/t/1600/480/1*rdVotAISnffB6aTzeiETHQ.png</sup></sup></sub></sup></sub></center>

# However, one thing we know that appears to be _fairly_ `general purpose` is a `biological brain` . On the other hand it really is not as it might be applicable to many domains, but is _heavily_ focused on a few `tasks` within the world it is surrounded by. This is also the idea what we should do with `AI` and referred to as the `AI set`.

# **The AI set**
# 
# | idea & definition | graphical representation |
# |-|-|
# |</br> </br> </br> This refers to a set of `tasks` animals are good at and perhaps some they are not good at due to physical limitations. </br> These `functions` are some `AI` should `learn` and be capable of accomplishing.|<img src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/AI_set.png" alt="logo" title="Github" width="500" height="280" />|
# 

# And this is exactly what we're going to talk about.

# ### Aim(s) of this section
# 
# - get to know the "lingo" and basic vocabulary
# - define important terms
# - situate core workflow aspects

# ### Outline for this section
# 
# 1. The definitions
# 2. The fellowship of core aspects
# 3. The two (or more) parts of each aspect
# 4. The return of the complexity 
# 5. Yes, it's that "easy" 

# ### The definitions
# 
# As with many other things it's important to define central terms and aspects. Here we're going to do that for `artificial intelligence`, `machine learning` and `deep learning`.
# 
# <img align="right" src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/AI.png" alt="logo" title="Github" width="320" height="120" />
# 
# **Artificial intelligence (AI)** is [intelligence](https://en.wikipedia.org/wiki/Intelligence) demonstrated by [machines](https://en.wikipedia.org/wiki/Machine), as opposed to the natural intelligence [displayed by humans](https://en.wikipedia.org/wiki/Human_intelligence) or [animals](https://en.wikipedia.org/wiki/Animal_cognition). Leading AI textbooks define the field as the study of ["intelligent agents"](https://en.wikipedia.org/wiki/Intelligent_agent): any system that perceives its environment and takes actions that maximize its chance of achieving its goals. Some popular accounts use the term "artificial intelligence" to describe machines that mimic "cognitive" functions that humans associate with the [human mind](https://en.wikipedia.org/wiki/Human_mind), such as "learning" and "problem solving", however this definition is rejected by major AI researchers.
# 
# [https://en.wikipedia.org/wiki/Artificial_intelligence](https://en.wikipedia.org/wiki/Artificial_intelligence)
# 
# 

# 
# <img align="right" src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/AI_ML.png" alt="logo" title="Github" width="320" height="120" />
# 
# **Machine learning (ML)** is the study of computer [algorithms](https://en.wikipedia.org/wiki/Algorithm) that can improve automatically through experience and by the use of data. It is seen as a part of [artificial intelligence](https://en.wikipedia.org/wiki/Artificial_intelligence). Machine learning algorithms build a model based on sample data, known as ["training data"](https://en.wikipedia.org/wiki/Training_data), in order to make predictions or decisions without being explicitly programmed to do so. A subset of machine learning is closely related to [computational statistics](https://en.wikipedia.org/wiki/Computational_statistics), which focuses on making predictions using computers; but not all machine learning is statistical learning. The study of [mathematical optimization](https://en.wikipedia.org/wiki/Mathematical_optimization) delivers methods, theory and application domains to the field of machine learning. [Data mining](https://en.wikipedia.org/wiki/Data_mining) is a related field of study, focusing on [exploratory data analysis](https://en.wikipedia.org/wiki/Exploratory_data_analysis) through [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning). Some implementations of machine learning use data and [neural networks](https://en.wikipedia.org/wiki/Neural_networks) in a way that mimics the working of a biological brain.
# 
# [https://en.wikipedia.org/wiki/Machine_learning](https://en.wikipedia.org/wiki/Machine_learning)
# 

# <img align="right" src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/AI_ML_DL.png" alt="logo" title="Github" width="320" height="120" />
# 
# **Deep learning** (also known as deep structured learning) is part of a broader family of [machine learning](https://en.wikipedia.org/wiki/Machine_learning) methods based on [artificial neural networks](https://en.wikipedia.org/wiki/Artificial_neural_networks) with [representation learning](https://en.wikipedia.org/wiki/Representation_learning). Learning can be [supervised](https://en.wikipedia.org/wiki/Supervised_learning), [semi-supervised](https://en.wikipedia.org/wiki/Semi-supervised_learning) or [unsupervised](https://en.wikipedia.org/wiki/Unsupervised_learning). [Artificial neural networks (ANNs)](https://en.wikipedia.org/wiki/Artificial_neural_network) were inspired by information processing and distributed communication nodes in [biological systems](https://en.wikipedia.org/wiki/Biological_system). ANNs have various differences from biological [brains](https://en.wikipedia.org/wiki/Brain). Specifically, neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analogue. The adjective "deep" in deep learning refers to the use of multiple layers in the network. Early work showed that a linear [perceptron](https://en.wikipedia.org/wiki/Perceptron) cannot be a universal classifier, but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can. 
# 
# 
# [https://en.wikipedia.org/wiki/Deep_learning](https://en.wikipedia.org/wiki/Deep_learning)
# 
# 

# ### The fellowship of core aspects
# 
# After outlining the major terms, we also need to define further concepts and vocabulary that is commonly used. We will do that based on a simplified graphical description which we will use throughout most of this part of the course. So, let's start with what a `model` refers to. 
# 
# 
# <center><img src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/core_aspects.png" alt="logo" title="Github" width="500" height="280" /></center>

# | Term         | Definition | 
# |--------------|:-----:|
# | Model |  A set of parameters that makes a prediction based on a given input. The parameter values are fitted to available data. |

# But what does `input`, `prediction` and `data` mean here? The `input` entails the `data` and the `prediction` in this case entails the `output`. Let's add the respective information to our graphical description.
# 
# <center><img src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/core_aspects_input_output.png" alt="logo" title="Github" width="500" height="280" /></center>

# In the world of `artificial intelligence` (in theoretic but also practical content) you will commonly see `X` being used to denote the `input` and `y` denoting the `output`. 
# 
# <center><img src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/core_aspects_x_y.png" alt="logo" title="Github" width="500" height="280" /></center>

# ### The two (or more) parts of each aspect
# 
# Importantly, each aspect can and has to be described and defined even further. For example, the `input` is usually outlined as a function of `samples` and `features`. 
# 
# 
# <center><img align="center" src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/core_aspects_features.png" alt="logo" title="Github" width="500" height="280" /></center>

# | Term         | Definition | 
# |--------------|:-----:|
# | Input |  Data from which we would like to make predictions. For example, results from lab tests (e.g. hematocrit, protein concentrations, response time) to inform a diagnosis (e.g. anemia, Alzheimer's, Muliple Sclerosis). Data is typically multidimensional, with each sample having multiple values that we think might inform our predictions. Conventionally, the dimensions of data are [number of subjects] x [number of features] |
# | Sample | One dimension of the `input` `data` which corresponds to a particular subset or part of the entire `input` `data`. For example, a `sample` could refer to a `person`, `condition` or `trial`  
# | Feature | One dimension of the `input` `data` which corresponds to a particular `measure`. When describing a `person`, `height`, `weight`, `hair colour` could be considered as `features`. |
# 	

# Obviously, the `output` can and needs to be further described/defined as well. Here, we could for example do that based on a function of `samples` and `labels`. 
# 
# <center><img src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/core_aspects_labels.png" alt="logo" title="Github" width="600" height="280" /></center>

# | Term         | Definition | 
# |--------------|:-----:|
# | Labels |  True values corresponding to an `input` `data` `sample` that we would like to accurately `predict`.  Also known as `target`|
# | Prediction |  The `output` of a `model` for a given `sample`. In the ideal case, the `prediction` should be the `sample`'s `label`.|

# Let's put all of this more into the perspective of `neuroscience`.
# 
# <center><img src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/core_aspects_examples.png" alt="logo" title="Github" width="600" height="340" /></center>

# ### The return of the complexity
# 
# Having outlined the core components of `artificial intelligence` workflows, `input`, `model` and `output`, we also need to talk about the many other things that are crucial. This comprises aspects situated within the core components but also "between" them. Regarding the latter, an operation commonly utilized is `preprocessing`.
# 
# <center><img src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/core_aspects_preprocessing.png" alt="logo" title="Github" width="600" height="340" /></center>

# | Term         | Definition | 
# |--------------|:-----:|
# | Preprocessing |  Change the `distribution`/`expression`/`scaling` of the `input` (`raw feature vectors`) to make it more suitable for `models`, e.g. via [standardization](https://en.wikipedia.org/wiki/Standardization), [scaling](https://en.wikipedia.org/wiki/Scaling) or [transforming](https://medium.com/vickdata/four-feature-types-and-how-to-transform-them-for-machine-learning-8693e1c24e80). |

# Going further concerning `models`, important aspects one needs to consider there are the `estimator` and the `complexity`.
# 
# <center><img src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/core_aspects_estimators_complexity.png" alt="logo" title="Github" width="600" height="340" /></center>

# | Term         | Definition | 
# |--------------|:-----:|
# | Estimator |  An instance of a `model`, specifically in `sklearn`. An `estimator` refers to a common interface in `sklearn` that is used to `train` and `evaluate` `models`. |
# | Complexity |  Refers to the number of `parameters` in a `model`. A `model` with `10 parameters` is said to be more `complex` than a `model` with `3`. |

# Finally (at least within our short and simplified endeavour here), we need to talk about the `metric` that can be situated between the `model` and the `output`. 
# 
# <center><img src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/core_aspects_metric.png" alt="logo" title="Github" width="600" height="340" /></center>

# | Term         | Definition | 
# |--------------|:-----:|
# | Metric |  A `function` that defines how good a `prediction` is.|

# Sounds like a lot, eh? Let's see how these things look like in action...

# ### Yes, it's that "easy"
# 
# Let's imagine we want to use the functional connectivity between brain regions to predict the age of human participants. Using `python` and a few of its fantastic packages, here [pandas](https://pandas.pydata.org/) and [sklearn](https://scikit-learn.org/stable/), we can make this analysis happen in no time.
# 
# At first we get our data:

# In[2]:


import urllib.request

url = 'https://www.dropbox.com/s/v48f8pjfw4u2bxi/MAIN_BASC064_subsamp_features.npz?dl=1'
urllib.request.urlretrieve(url, 'MAIN2019_BASC064_subsamp_features.npz')


# Next, we load our input and inspect it:

# In[4]:


import numpy as np

data = np.load('MAIN2019_BASC064_subsamp_features.npz')['a']
data.shape


# We can also use [plotly]() to easily visualize our input data in an interactive manner (please click on the `+` to see the `code`):

# In[4]:


import plotly.express as px
from IPython.core.display import display, HTML
from plotly.offline import init_notebook_mode, plot

fig = px.imshow(data, labels=dict(x="features", y="participants"), height=800, aspect='None')

fig.update(layout_coloraxis_showscale=False)
init_notebook_mode(connected=True)

#fig.show()

plot(fig, filename = '../../../static/input_data.html')
display(HTML('../../../static/input_data.html'))


# To get a better idea of what this means, let's have a look at the feature from the neuroscience perspective, ie. the `brain network` (please click on the `+` to see the `code`):

# In[40]:


from nilearn import datasets
data = datasets.fetch_development_fmri(n_subjects=1)
parcellations = datasets.fetch_atlas_basc_multiscale_2015(version='sym')
atlas_filename = parcellations.scale064

from nilearn.input_data import NiftiLabelsMasker

masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, 
                           memory='nilearn_cache', verbose=1)


time_series = masker.fit_transform(data['func'][0])

from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = np.squeeze(correlation_measure.fit_transform([time_series]))


import numpy as np
# Mask the main diagonal for visualization:
#np.fill_diagonal(correlation_matrix, 0)

# The labels we have start with the background (0), hence we skip the
# first label
from nilearn.plotting import find_parcellation_cut_coords, view_connectome

coords = find_parcellation_cut_coords(atlas_filename)

view_connectome(correlation_matrix, coords, edge_threshold='80%', 
                title='Features displayed from a network perspective', colorbar=False)


# Beside the `input data` we also need our `labels`:

# In[5]:


url = 'https://www.dropbox.com/s/ofsqdcukyde4lke/participants.csv?dl=1'
urllib.request.urlretrieve(url, 'participants.csv')


# Which we can easily load and check via [pandas]():

# In[6]:


import pandas as pd
labels = pd.read_csv('participants.csv')['AgeGroup']
labels.describe()


# For a better intuition, we're going to also visualize the labels and their distribution (please click on the `+` to see the `code`): 

# In[7]:


fig = px.histogram(labels, marginal='box', template='plotly_white')

fig.update_layout(showlegend=False, width=800, height=600)
init_notebook_mode(connected=True)

#fig.show()

plot(fig, filename = '../../../static/labels.html')
display(HTML('../../../static/labels.html'))


# And we're ready to create our machine learning analysis [pipeline]() using [sklearn]() within we will [scale]() our input data, train a [Support Vector Machine]() and test its [predictive performance](). We import the required `functions` and `classes`:

# In[8]:


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# and setup a [sklearn pipeline]():

# In[9]:


pipe = make_pipeline(StandardScaler(), SVC())


# After dividing our input and labels into [training]() and [test]() sets:

# In[10]:


X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=0)


# we can already [fit]() our machine learning analysis `pipeline` to our data: 

# In[11]:


pipe.fit(X_train, y_train)


# and test its predictive performance:

# In[12]:


print('accuracy is %s with chance level being %s' 
      %(accuracy_score(pipe.predict(X_test), y_test), 1/len(pd.unique(labels))))


# - but wait...there's much more to talk about here

# **You lied!**
# 
# <center><img src="https://c.tenor.com/jXn6jy2JkosAAAAC/the-lies-rage.gif" alt="logo" title="Github" width="400" height="280" /></center>
# <sub><sup><sub><sup><sup>https://c.tenor.com/jXn6jy2JkosAAAAC/the-lies-rage.gif
# </sup></sup></sub></sup></sub>

# **I'm sorry.**
# 
# <center><img src="https://media4.giphy.com/media/sS8YbjrTzu4KI/giphy.gif?cid=ecf05e47ygr63lz155vgnqurenhc4hoizy9hhcw8dfxtqi2f&rid=giphy.gif&ct=g" alt="logo" title="Github" width="400" height="280" /></center>
# <sub><sup><sub><sup><sup>https://media4.giphy.com/media/sS8YbjrTzu4KI/giphy.gif?cid=ecf05e47ygr63lz155vgnqurenhc4hoizy9hhcw8dfxtqi2f&rid=giphy.gif&ct=g
# </sup></sup></sub></sup></sub>

# The truth is: while it's very easy (maybe too easy?) to setup and run machine learning analyses, doing it right is definitely not (like a lot of other complex analyses)... There are so many aspects to think about and so many things one can vary that the [garden of forking paths]() becomes tremendously large. Thus, we will spent the next few hours to go through central concepts and components of these analyses, first for ["classic" machine learning]() and then for [deep learning]().

# One aspect that is part of every `model` is the `fitting`, i.e. the `learning` of `model weights` (`machine learning` much?. Thus, we need to go through some things there as well...

# ### Model fitting
# 
# - when talking about `model fitting`, we need to talk about three central aspects:
#     - the model
#     - the loss function
#     - the optimization

# | Term         | Definition | 
# |--------------|:-----:|
# | Model |  A set of parameters that makes a prediction based on a given input. The parameter values are fitted to available data.|
# | Loss function | A function that evaluates how well your algorithm models your dataset |
# | Optimization | A function that tries to minimize the loss via updating model parameters. |
# 	

# #### An example: linear regression
# 
# - Model:  $$y=\beta_{0}+\beta_{1} x_{1}^{2}+\beta_{2} x_{2}^{2}$$
# - Loss function: $$ M S E=\frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right)^{2}$$
# - optimization: [Gradient descent]()
# 

# - `Gradient descent` with a `single input variable` and `n samples`
#     - Start with random weights (`β0` and `β1`) $$\hat{y}_{i}=\beta_{0}+\beta_{1} X_{i}$$
#     - Compute loss (i.e. `MSE`) $$M S E=\frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right)^{2}$$
#     - Update `weights` based on the `gradient`
#     
# <center><img src="https://cdn.hackernoon.com/hn-images/0*D7zG46WrdKx54pbU.gif" alt="logo" title="Github" width="550" height="280" /></center>
# <sub><sup><sub><sup><sup>https://cdn.hackernoon.com/hn-images/0*D7zG46WrdKx54pbU.gif
# </sup></sup></sub></sup></sub>
# 

# - `Gradient descent` for complex models with `non-convex loss functions`
#     - Start with random weights (`β0` and `β1`) $$\hat{y}_{i}=\beta_{0}+\beta_{1} X_{i}$$
#     - Compute loss (i.e. `MSE`) $$M S E=\frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right)^{2}$$
#     - Update `weights` based on the `gradient`
#     
# <center><img src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/gradient_descent_complex_models.png" alt="logo" title="Github" width="500" height="280" /></center>

# ### Questions
# 
# <center><img src="https://media.giphy.com/media/xT5LMB2WiOdjpB7K4o/giphy.gif?cid=ecf05e47y84d0uuf88r7mrmnj56d3gyvycf8yhdunfyqamhx&rid=giphy.gif&ct=g" alt="logo" title="Github" width="400" height="280" /></center>
# <sub><sup><sub><sup><sup>https://media.giphy.com/media/xT5LMB2WiOdjpB7K4o/giphy.gif?cid=ecf05e47y84d0uuf88r7mrmnj56d3gyvycf8yhdunfyqamhx&rid=giphy.gif&ct=g
# </sup></sup></sub></sup></sub>
# 
# 
