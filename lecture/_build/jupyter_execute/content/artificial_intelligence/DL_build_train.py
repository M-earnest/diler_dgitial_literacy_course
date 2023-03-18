#!/usr/bin/env python
# coding: utf-8

# # Build and train your neural network

# ### Aim(s) for this section 🎯
# 
# - get practical experience with `ANN`s, specifically `CNN`s
# - `built`, `train` and `evalaute` a `CNN`
# - discuss important building blocks and learn how to interpret outcomes

# ### Outline for this section 📝
# 
# 1. The tutorial dataset
#     - preparing the data
# 2. `Building` and `training` an `ANN` - a `2D CNN` example
#     - `python` and `deep learning`
#     - defining the basics
#     - building an `ANN`
#     - how to train your network
# 3. `Evaluating` an `ANN`
#     - The `test set`
#     - `Confusion matrix`
#     - `Generalization`
#     - `Transfer learning`

# In[1]:


import random
random.seed(0)


# ### The tutorial dataset
# 
# In order to demonstrate how you can `build` and `train` an `ANN` we need a dataset that fits several requirements:
# 
# - we are all here with `laptops` that most likely don't have the computational power of `HPC`s and graphic cards (if you do, good for you!), thus the dataset needs to be small enough so that we can actually train our `ANN` within a short amount of time and without `GPU`s
# - thinking this further: we also might not want to test the most simplest `ANN`, but one with a few `hidden layers`
# - it would be cool to use a `dataset` with at least some real world feeling to demonstrate a somewhat typical workflow

# We thus decided on a small `fMRI` dataset from [Zhang et al.](https://link.springer.com/article/10.1007%2Fs12021-013-9187-0) with the following specs:
# 
# - two [resting-state]() sessions from `48` participants
# - one with `eyes-closed` and one with `eyes-open`
# - we will use a subset of `volumes` of each session
# 
# <img align="right" src="https://github.com/miykael/workshop_pybrain/blob/master/workshop/notebooks/data/resting_state_eyes.gif?raw=true" alt="logo" title="Github" width="400" height="280" />

# This will allow us to:
# 
# - address a (somewhat) realistic `image processing` task via `supervised learning` for which we can employ a `CNN`
# - showcase how parameters might change the `ANN`
# - evaluate `representations` across `layers`

# A note on the datasets utilized here:
# 
# - we're very sorry that it's so `(f)MRI` focused
# - we tried to include other modalities, specifically microscopy, but:
#     - couldn't find datasets that fit the setup and framework of the workshop
#     - don't have enough experience with this modality to adapt existing ones
# - however, we collected a few resources on `machine` and `deep learning` for microscopy [here]()
#     - contain a variety of pre-trained models 
#     - info on how to prepare data
# - we also tested and checked a few things in advance so that we can
#   help you during the hands-on in the best way possible    

# <img align="center" src="https://media4.giphy.com/media/rvDtLCABDMaqY/giphy.gif?cid=ecf05e47c5qyer72l87resjeadw2zu6kdoqq1b6guo8gqr9d&rid=giphy.gif&ct=g" alt="logo" title="Github" width="400" height="280" />
# <sub><sup><sub><sup><sup>https://media4.giphy.com/media/rvDtLCABDMaqY/giphy.gif?cid=ecf05e47c5qyer72l87resjeadw2zu6kdoqq1b6guo8gqr9d&rid=giphy.gif&ct=g
# </sup></sup></sub></sup></sub>

# - that being said, let's gather our `dataset`

# In[2]:


import urllib.request

url = 'https://github.com/miykael/workshop_pybrain/raw/master/workshop/notebooks/data/dataset_ML.nii.gz'
urllib.request.urlretrieve(url, 'dataest_ML.nii.gz')


# - and check its dimensions as well as visually inspect it:

# In[3]:


import nibabel as nb


# In[4]:


data = nb.load('dataest_ML.nii.gz')


# In[5]:


data.shape


# In[6]:


data.orthoview()


# We can also plot the `mean image` across `time` to get an idea about `signal variation`:

# In[7]:


from nilearn.image import mean_img
from nilearn.plotting import view_img


# In[8]:


view_img(mean_img(data))


# Well well well, there should be something in there that an `ANN` can learn...

# - the task:
# 
#     - we know that there are `images` where participants had their `eyes open` or `closed`
#     - we now want to `build` an `ANN` to `train` it to recognize and distinguish the respective `images`
#     - we also want to know what representations our `ANN` learns
#     - thus, we have a `supervised learning problem` which we want to solve via `image processing`
#     

# - what we need to do:
#     - prepare the data
#     - decide on a model, build and train it

# **Preparing the data**
# 
# From our adventures in `"classic" machine learning` we know, that we need `labels` to address a `supervised learning problem`. Checking the dimensions of our `dataset` again:

# In[9]:


data.shape


# We see that we have a `4 dimensional dataset`, with the first three dimensions being spatial, i.e. `x`, `y` and `z`, and the fourth being time. So we need to specify during which of the `images` participants had their `eyes closed` and during which they had their `eyes open`. Without going into further detail, we know that it's always `4 volumes` of `eyes closed`, followed by `4 volumes` of `eyes open`, etc. and given that we have `48` participants, we can define our `labels` as follows:

# In[10]:


import numpy as np

labels = np.ravel([[['closed'] * 4, ['open'] * 4] for i in range(48)])
labels[:20]


# Going back to the aspect of `computation time` and `resources`, as well as given that this is a showcase, it might be a good idea to not utilize the entire `fMRI volume`, but only certain parts where we expect some things to happen. (Please note: this is of course a form of `inductive bias` comparable to `feature engineering` in `"classic" machine learning` and something you won't do in a "real-world situation" (depending on the data and goal of course)).

# In our case, we could try to not train the neural network only on one very thin slab (a few slices) of the brain. So, instead of taking the data matrix of the whole brain, we just take 2 slices in the region that we think is most likely to be predictive for the question at hand.
# 
# We know (or suspect) that the regions with the most predictive power are probably somewhere around the eyes and in the visual cortex. So let's try to specify a few slices that cover those regions.

# So, let's try to just take a few slices around the eyes:

# In[11]:


from nilearn.plotting import plot_img
plot_img(mean_img(data).slicer[...,5:-25], cmap='magma', colorbar=False,
          display_mode='x', vmax=2, annotate=False, cut_coords=range(0, 49, 12),
          title='Slab of the mean image');


# This worked only so and so, but with a few lines of code the mighty power of `python` and its `packages` can help us achieve a better `training dataset`. For example, we could rotate the `volume` (depending on the data and goal, this sort of image processing is actually sometimes done in "real-world situations"):

# In[12]:


# Rotation parameters
phi = 0.35
cos = np.cos(phi)
sin = np.sin(phi)

# Compute rotation matrix around x-axis
rotation_affine = np.array([[1, 0, 0, 0],
                            [0, cos, -sin, 0],
                            [0, sin, cos, 0],
                            [0, 0, 0, 1]])
new_affine = rotation_affine.dot(data.affine)


# Now we can use this new `affine` to `resample` our `volumes`:

# In[13]:


from nilearn.image import resample_img
new_img = nb.Nifti1Image(data.get_fdata(), new_affine)
img_rot = resample_img(new_img, data.affine, interpolation='continuous')


# How do our `volumes` look now? 

# In[14]:


plot_img(mean_img(img_rot).slicer[...,5:-25], cmap='magma', colorbar=False,
          display_mode='x', vmax=2, annotate=False, cut_coords=range(0, 49, 12),
          title='Slab of the mean rotated image');


# Coolio! Now we can check what set of `slices` of our `volumes` might constitute feasible inputs to our `ANN`: 

# In[15]:


from nilearn.plotting import plot_stat_map
img_slab = img_rot.slicer[..., 12:15, :]
plot_stat_map(mean_img(img_slab), cmap='magma', bg_img=mean_img(img_slab), colorbar=False,
              display_mode='x', vmax=2, annotate=False, cut_coords=range(-20, 30, 12),
              title='Slices of the rotated image');


# Now this is something we can definitely work with, even if we have only limited time and resources.

# ### Building and training an `ANN` - a `2D CNN` example
# 
# Not that we have checked and further prepared our `dataset`, it's finally time to get to work. Given that we're working with `fMRI volumes`, i.e. `images` and what we've heard about the different `ANN architectures`, using a `CNN`  might be a good idea. 

# But where to start? Is there any software I can use that makes the building, training and evaluating of `ANN`s "comparably easy"?

# Well, say no more...`Python` obviously also has your back when it's about `deep learning` (gotta love `python`, eh?)! It actually has not only but a bunch of different packages that focus on `deep learning`. Let's have a brief look on the things that are out there. 

# #### Python and `deep learning`
# 
# As outlined before `python` is a very powerful `all purpose language`, including a broad user base and support for `machine learning`, both "classic" and `deep learning`. 
# 
# 
# <img align="center" src="https://miro.medium.com/max/1400/1*RIrPOCyMFwFC-XULbja3rw.png
# " alt="logo" title="Github" width="400" height="280" />
# <sub><sup><sub><sup>https://miro.medium.com/max/1400/1*RIrPOCyMFwFC-XULbja3rw.png
# </sup></sub></sup></sub>

# - lots of well `documented` and `tested` `libraries`
# - lots of `tutorials` to learn things (you + the `ANN`):
#     - youtube videos
#     - blog posts
#     - other open workshops
#     - jupyter notebooks
# - lots of `pre-trained models` to use for your research    
# - lots of support in forums
# - completely free and open source!

# <img align="center" src="https://miro.medium.com/max/700/1*s_BwkYxpGv34vjOHi8tDzg.png" alt="logo" title="Github" width="400" height="280" />
# <sub><sup><sub><sup>https://miro.medium.com/max/700/1*s_BwkYxpGv34vjOHi8tDzg.png
# </sup></sub></sup></sub>
# 
# 

# - all work a bit different, but the basic concepts and steps are comparable
#     - nevertheless: always check the documentation as e.g. `default values` might vary
# - crucial in all: [tensors](https://en.wikipedia.org/wiki/Tensor)
#     - have a look at this great [introduction to tensors from tensorflow](https://www.tensorflow.org/guide/tensor)

# - the question which one to choose is of course not an easy one and might also depend on external factors:
#     - the type and amount of data you have
#     - the time and computational resources available to you
#     - specific functionality that only exists in a certain package
#     - utilization of pre-trained `ANN`s
#     - what you've heard about and others show you (that's obviously on us...)

# - here we will use [keras](https://keras.io/) which is build on top of [tensorflow](https://en.wikipedia.org/wiki/TensorFlow) because:
#     - high-level `API`
#     - easy to grasp implementation of `ANN` building blocks
#     - fast experimentation
# - for a fantastic resource that includes all things we talked about/will talk and way more in much greater detail, please check the [deep learning part of Neuromatch Academy](https://deeplearning.neuromatch.io/tutorials/intro.html)

# - important: we're not saying that `keras`/`tensorflow` is better than the other `python deep learning libraries`, it just works very well for tutorials/workshops like the one you're currently at given the very limited time we have

# Now it's finally go time, get your machines ready!
# 
# <img align="center" src="https://c.tenor.com/1cbzhT0TKTMAAAAd/cat-asleep.gif" alt="logo" title="Github" width="400" height="280" />
# <sub><sup><sub><sup><sup>https://c.tenor.com/1cbzhT0TKTMAAAAd/cat-asleep.gif
# </sup></sup></sub></sup></sub>

# #### Defining the basics
# 
# Before we can actually assemble our `ANN`, we need to set a few things. However, first things first: `importing` `modules` and `classes`:

# In[16]:


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, AvgPool2D, BatchNormalization
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam


# Next, we need to take a look at our `dataset` again, specifically its `dimensions`:

# In[17]:


img_slab.shape


# Again, we have the `x`, `y` and `z` of our `images`, i.e. the `images` themselves, in the first three dimensions which are stacked in the fourth dimension. For this type of `data` to work with `keras`/`tensorflow` we actually need to adapt, that is swap, some of the dimensions, as these `modules`/`functions` expect them in a different way. This part of getting your `data` ready as `input` into a given `ANN` is crucial and can cause one or the other problem. Therefore, always make sure to carefully read the `documentation` of a `class`, `module` or `pre-trained model` you want to use. They are usually very good and show entail examples of how to get data ready for the `ANN`.  

# That being said, here we need basically only need to make the last `dimension` the first, so that we have the `volumes`/`images` stacked in the first `dimension` and the `images` themselves within the subsequent three:  

# In[18]:


data = np.rollaxis(img_slab.get_fdata(), 3, 0)
data.shape


# Specifically, the last `dimension`, here `3`, are considered as `channels`.

# There are some central `parameters` we can set before building the `ANN` itself. For example, we know the shape of the `input`. That is, the `dimensions` our `input layer` will receive: 

# In[19]:


data_shape = tuple(data.shape[1:])
data_shape


# We also want to set the `kernel size` of our `convolutional kernel`. As heard before, this can be a tremendously important `hyperparamter` that can drastically affect the behavior of your `ANN`. It is thus something you have to carefully think about and even might want to evaluate via `cross-validation`. Here, we will use a `kernel size` of `(3,3)`.

# In[20]:


kernel_size = (3, 3)


# The same holds true for the `filters` we want our `convolutional layers` to use:  

# In[21]:


filters = 32


# Given that we want to work with a `supervised learning problem` and know that there are `2 classes` we want our `ANN` to learn to learn distinguish, we can set the number of `classes` accordingly: 

# In[22]:


n_classes = 2


# With that, we ready to start building our `ANN`!

# #### Building an `ANN`
# 
# You heard right, it's finally `ANN` time! Initially, we have to decide on an `architecture`, that is the type of `ANN` we want to build. As we want to test a simple `CNN`, a `feedforward ANN` without multiple `inputs` and/or `outputs`, we will employ what is called a `sequential model` in `keras`/`tensorflows` within which we define `layer` by `layer`. Note: It's the easiest but also the most restrictive one.  

# In[23]:


model = Sequential()


# Now that the basic structure is defined, we can start `adding` `layers` to our `ANN`. This is achieved by the following syntax (`pseudocode`):
# 
# ``model.add(layer_typ(layer_settings, layer_parameters))``

# ##### Defining the `input layer`
# 
# The first step? Obviously defining an `input layer`, i.e. the `layer` that receives the `external input`. We want to build a `CNN`, so let's make it a `convolutional layer`. What do we need for that?

# In[24]:


help(Conv2D)


# Ok, there are quite a few `parameters` to set. However, we are going to keep it light and breezy, setting a few of the things we've talked about: the number of `filter`, the `kernel size`, the `activation function` and the `shape of the input` which in our case is the shape of our `images`.

# In[25]:


model.add(Conv2D(filters, kernel_size, activation='relu', input_shape=data_shape))


# ##### Batch normalization layer
# 
# As briefly addressed before, [batch normalization](https://en.wikipedia.org/wiki/Batch_normalization) can be very helpful: speed up the training, addresses internal covariate shift (highly debated), smoothes the loss function, etc. . It does so via `re-centering` and `re-scaling` the inputs of a given `layer`. Thus, we are going to include `batch normalization layers` also in our `ANN`:

# In[26]:


model.add(BatchNormalization())


# As you can see, we added the `batch normalization layer` right after the `convolutional layer` so that the latter's output will be `re-centered` and `re-scaled`.

# ##### Pooling layer
# 
# Another important part of `CNN architectures` is the [`pooling layer`](https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer), i.e. the `layer` that reduces the `spatial size` of the `representation` computed in the `previous layer`, i.e. `convolutional layer`. In turn, we can reduce the amount of `parameters` and thus computation our `ANN` needs to perform. Out of the two pooling options, `max pooling` and `average pooling`, `CNN`s typically utilize `max pooling` because it helps to detect certain `features` more easily and as the `representation` becomes more `abstract` also helps to reduce `overfitting`. Sounds like a good idea, eh? 

# In[27]:


model.add(MaxPooling2D())


# ##### Getting more fine-grained
# 
# In order to get our `ANN` and the `features` it works on more fine-grained, we will double the `filter size` for the next step, i.e. `layer`(s).

# In[28]:


filters *= 2


# Along this line of thought, we will repeat the succession of `convolutional`, `batch normalization`, `pooling` and `filter size increase` two more times:

# In[29]:


model.add(Conv2D(filters, kernel_size, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
filters *= 2

model.add(Conv2D(filters, kernel_size, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
filters *= 2


# Please note: we removed the `input_shape` parameter from the `Conv2D layers` as they are not `input layers`. 

# ##### It's getting dense
# 
# Now that we've sent our `input` through several `layers` aimed at obtaining `representations`, it might be worth a try to think about we can achieve our `supervised learning` goal. Given that we want to have a `binary outcome`, i.e. `eyes open` or `eyes closed`, we want to `classify`. We can achieve this via [dense or fully connected layers](https://en.wikipedia.org/wiki/Convolutional_neural_network#Fully_connected_layers) (think about `MLP`s again). However, for this to work, we need to add a [flatten layer](https://keras.io/api/layers/reshaping_layers/flatten/) before that. The reason: even though we `convoluted` and `pooled` our input quite a bit, it's still `multidimensional` and we need it `linear` to pass it through a `dense/fully connected layer`. 

# In[30]:


model.add(Flatten())


# Another thing we need to remember is [regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)), that is we need to address `overfitting`. A brief recap: given that our `ANN` will have a large number of `parameters` together with the `universal function approximation theorem`, there's for example the possibility that our `ANN` will just "memorize" the `dataset` without capturing the `information` we want to obtain, thus failing to `generalize` to new `data`. And why that's cool in theory (the `memorizing` part, not the failed `generalization` part), we obviously want to avoid that. Therefore, we need to apply `regularization` via imposing `constraints` on the `ANN`'s `parameters` or adapting the `cost function`. One way to go would be the application of [dropout layers](https://keras.io/api/layers/regularization_layers/dropout/) that `randomly` and `temporally` set `nodes` in our `layers` to `0`, i.e. `deleting` them during the `training`.

# In[31]:


model.add(Dropout(0.5))


# The `parameter` we added here, `0.5`, specifies the `dropout rate` or in other words the fraction of the `input units`, i.e. `nodes`, to drop. This is a commonly applied value, but does not mean it should also be the `default`!

# Time to go `dense` and start with our first respective `layer`. As with the other `layer` types, there a bunch of `parameters` we can define:

# In[32]:


help(Dense)


# For now, we will focus on the `output size`/`dimensionality` of the `layer`, the `activation function`, as well as the `kernel` and `bias initializers`. As every `node` in a `dense/fully connected layer` will receive `input` from all `nodes` of the `previous layer` one of its drawbacks is `computation time` based on the amount of `parameters`. However, due to its underlying `matrx-vector multiplication` and output in _n_ `dimensional` vectors, we can use it to change the `dimensions` of the `vector`, downscaling it from the `multidimensional input` it receives from the `convolutional layer(s)`. Here, we will set it to `1024`. The `activation function` might be old news to you now, but just to be sure: our `dense/fully connected layer` will have a `non-linear activation function`, specifically, `ReLu`. Based on that, we can also choose a `kernel initializer` that is `optimized` for this `activation function`: [Kaming He initialization](https://arxiv.org/pdf/1502.01852). The `bias initializers` will be set to `zeros` following common practice backed up by various studies.

# In[33]:


model.add(Dense(1024, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros'))


# Before we go to the next `dense/fully connected layer`, we will integrate a few of the things we talked about again. Namely, `batch normalization` and `dropout layers`.

# In[34]:


model.add(BatchNormalization())
model.add(Dropout(0.5))


# To further reduce the number of `dimensions` for our final, i.e. the `output`, `layer`, we will a create a short succession as we've done with the `convolutional layers` via repeating the `dense/fully connected - batch normalization - dropout layer` sequence two times, each time reducing the `dimensions` of the `output` by a factor of `4`: 

# In[35]:


model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


# ##### It's the final countdown....sorry, layer
# 
# We've reached the end of our `ANN`, the `output layer`. Again: we are confronted with a `supervised learning problem` within which we want to train our `ANN` to perform a `binary classification` between `eyes closed` and `eyes open`. Thus, the final `layer`, will be a `dense/fully connected layer` again, which has as many `outputs` as we have classes: `2`. Additionally, we change the `activation function` to `softmax` so that we will obtain a `normalized probability distribution` with values ranging between `0` and `1`, indicating the probability of belonging to either `class`. These will then be compared to the `true labels`.

# In[36]:


model.add(Dense(n_classes, activation='softmax'))


# ##### There's a steep learning curve when you curve without learning
# 
# While all of this is definitely amazing and already hard to comprehend (at least for me), one, actually THE ONE, aspect of `machine/deep learning` is missing: we haven't told our `ANN` how it should `learn`. In more detail, we need to tell our `ANN` how it should compare the `probabilities` computed in the `output layer` to the `true labels` and `learn` via a `loss function` and an `optimizer` to minimize the respective `error`. Given our `learning problem` and `dataset`, we will go rather "classic" and use `accuracy` as our `metric`, `sparse_categorical_crossentropy` as our `loss function` and `adam` as our `optimizer`. Importantly, these `parameters` will be defined during the `compile` step which will finally `build` our `ANN`.

# In[37]:


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', 
              metrics=['accuracy'])


# We know how it feels...
# 
# <img align="center" src="https://c.tenor.com/NcibGDKTKQAAAAAd/status-tired.gif" alt="logo" title="Github" width="400" height="280" />
# <sub><sup><sub><sup><sup>https://c.tenor.com/NcibGDKTKQAAAAAd/status-tired.gif
# </sup></sup></sub></sup></sub>

# #####  A fresh start
# 
# We really dragged this one out, didn't we? Sorry folks, we thought it might be a good idea to really go step by step...To, however, maybe see everything at once, we will do a version with all the necessary `code` in one `cell`. 

# In[38]:


n_classes = 2

filters = 32

kernel_size = (3, 3)

model = Sequential()

model.add(Conv2D(filters, kernel_size, activation='relu', input_shape=data_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D())
filters *= 2

model.add(Conv2D(filters, kernel_size, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
filters *= 2

model.add(Conv2D(filters, kernel_size, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
filters *= 2

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(1024, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', 
              metrics=['accuracy'])


# Still, that's a lot. Isn't there an easy way to check things more conveniently?  

# There is or more precisely: there are, because of more than one option to do so.

# The first one is rather simple. Our `ANN` has a `.summary()` function which will provide us with a nice overview as well as details about its `architecture`. (This is also a great way to check out pre-trained models.)

# In[39]:


model.summary()


# We can nicely see all of our `layers` and their `dimensions`, as well as they change along the `ANN` and with them, the respective `representations`.

# One thing we haven't really talked about so far for our `ANN` but which becomes abundantly clear here: the high number of `parameters`: `2,345,602`. Kinda wild, isn't it? Especially considering that our `ANN` isn't "that complex". Others have waaaaaay more...
# 
# <img align="center" src="https://c.tenor.com/5ety3Lx3QccAAAAC/its-fine-dog-fine.gif" alt="logo" title="Github" width="400" height="280" />
# <sub><sup><sub><sup><sup>https://c.tenor.com/5ety3Lx3QccAAAAC/its-fine-dog-fine.gif
# </sup></sup></sub></sup></sub>

# Another cool option to inspect our `ANN` is to use [tensorboard](https://www.tensorflow.org/tensorboard/) which will evaluate after the next step.

# #### How to train your network
# 
# As some might say: this is where the real fun starts. We have built and checked our `ANN`. Now, it's time to let it `learn`. Comparable to the `models` we utilized in the first part of the workshop, the "classic" `machine learning` models, we need to fit it in order to `train` it. Or more accurately: to let it learn `representations` that are helpful to achieve its given `task`. Going back to the [previous section](), we discussed two important `parameters` we can define for this endeavor: the [epochs]() and the [batch size](). 

# A brief recap: 
# 
# - an [epoch]() refers to one cycle through the entire `training dataset`, i.e. our `ANN` went through the entire `training dataset` once. Thus, the number of `epochs` describes how often the `ANN` worked through the entire `training dataset` during the `fitting`.
# - a [batch]() refers to the number of `samples` the `ANN` goes through before it will update its `weights` based on the combination of `metric`, `loss function` and `optimizer`. Thus, the number of `batches` defines how often the `weights` are updated during an `epoch`. 

# Both `epoch` and `batch` are thus `parameters` for the `learning` and not `parameters` obtained by `learning`.

# In order to apply it to and understand it based on our example `dataset` we need to define a `training` and `test dataset` as we did before. (The same things about `training`, `testing` and `validating` we talked about during `"classic" machine learning` also hold true here.) We can use our old friend `scikit-learn` for this.

# We define our `y` based on our `labels`, simply converting it them to `true` for `eyes open` and `false` for `eyes closed`:

# In[40]:


y = labels =='open'
y.shape


# In[41]:


y[:10]


# With that we can split our `dataset` into `training` and `test` sets:

# In[42]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=0, shuffle=False)

print('Shapes of X:', X_train.shape, X_test.shape)
print('Shapes of y:', y_train.shape, y_test.shape)


# Ok, we got `307` `samples` in the `train` and `77` `samples` in the `test set`.

# Back to `epochs` and `batches`: if we set our `batch size` to e.g. `32` and our `epochs` to e.g. `125`, it would mean we have `32 batches` within each of `125 epochs`. So, the `ANN` would go through ~9 `images` (`307 test images/32 batches`, some `batches` will have more `images` than others) before updating its `weights`. Also, the `ANN` will go through the entire `training dataset` `125 times` and thus through `4000 batches`. Please note: while this sounds like a lot, the number of `epochs` is usually waaaaay higher, in the hundreds and thousands! However, once more: within our setting here and the `computional resources` we have, we have to keep it short. Additionally, determining the "correct" number of `batches` and `epochs` is far from being easy and may even present an ill-posed question. That being said, we will use the example sizes we went through.

# In[43]:


batch_size = 32


# In[44]:


nEpochs = 125


# Folks, it's finally time to `train` our `ANN` and let it `learn`. To keep of the things that happen, will set a few things so that we can utilize `tensorboard` later on. For this to work we need to load the respective [jupyter extension](https://www.tensorflow.org/tensorboard), define a directory to where we can save the `logs` of the `training` and the define the so-called `callback` which will be included in the `.fit()` function.

# In[45]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[ ]:


import datetime, os
import tensorflow as tf

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


# To kick things off, we use the `.fit` function of our `model` and start the `training`. 

# In[47]:


get_ipython().run_line_magic('time', 'fit = model.fit(X_train, y_train, epochs=nEpochs, batch_size=batch_size, validation_split=0.2, callbacks=[tensorboard_callback])')


# How does it feel, having `built` and `trained` your first `ANN`? Isn't it beautiful and wild? Seeing it in action after all this (hopefully not too terrible) theoretical content and preparation is definitely something else. Y'all obviously deserve to party for a minute!
# 
# 
# <img align="center" src="https://c.tenor.com/p6gcBayghrEAAAAC/baby-yoda.gif" alt="logo" title="Github" width="400" height="280" />
# <sub><sup><sub><sup><sup>https://c.tenor.com/p6gcBayghrEAAAAC/baby-yoda.gif
# </sup></sup></sub></sup></sub>
# 

# Ok, time to get back to work. We might have `built` and `trained` our `ANN`, but actually have no idea how did it perform during the `training`. There were some hints (actually all information we're interested in) in the output we saw during the training, but let's visualize it to better grasp it. We will start with the `metric`:

# In[48]:


import plotly.graph_objects as go
import numpy as np
from plotly.offline import plot
from IPython.core.display import display, HTML

epoch = np.arange(nEpochs) + 1

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=epoch, y=fit.history['accuracy'],
                    mode='lines+markers',
                    name='training set'))
fig.add_trace(go.Scatter(x=epoch, y=fit.history['val_accuracy'],
                    mode='lines+markers',
                    name='validation set'))

fig.update_layout(title="Accuracy in training and validation set",
                  template='plotly_white')

fig.update_xaxes(title_text='Epoch')
fig.update_yaxes(title_text='Accuracy')

#fig.show()

plot(fig, filename = 'acc_eyes.html')
display(HTML('acc_eyes.html'))


# Question: what do you see and how do you interpret it?

# After checking the `accuracy metric` of our `ANN`, we will have a look at the `loss function`.

# In[49]:


import plotly.graph_objects as go
import numpy as np

epoch = np.arange(nEpochs) + 1

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=epoch, y=fit.history['loss'],
                    mode='lines+markers',
                    name='training set'))
fig.add_trace(go.Scatter(x=epoch, y=fit.history['val_loss'],
                    mode='lines+markers',
                    name='validation set'))

fig.update_layout(title="Loss in training and validation set",
                  template='plotly_white')

fig.update_xaxes(title_text='Epoch')
fig.update_yaxes(title_text='Loss')

#fig.show()

plot(fig, filename = 'loss_eyes.html')
display(HTML('loss_eyes.html'))


# Question: what do you see and how do you interpret it?

# As promised, there's another option to check our `ANN` and its behavior: `tensorboard`, which we can finally bring up now that our `ANN` is trained. Beside also getting the above generated graphs, we get a `graph representation` of our `ANN`, `distributions` and `histograms` of our `batch normalization`, as well as detailed `time series` for many of these `values`. It is nothing but fantastic! (Unfortunately, this super cool feature won't work in the rendered `jupyter book`.)

# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir logs')


# We know how our `ANN` performed during the `training` and saw that it definitely `learned` something. How do we make sure, it actually `learned` meaningful `representations` and not "just" `memorized`? We need to test its `generalizability` by evaluating it on our `test set`!

# ### Evaluating an `ANN`
# 
# #### The test set 
# 
# Evaluating an `ANN` is not different from evaluating a "classic" `machine learning model`. We simply task it to perform its given `task` on the hold-out `test set`. Here this is done via the `.eval()` function of our `trained model`:

# In[51]:


evaluation = model.evaluate(X_test, y_test)
print('Loss in Test set:      %.02f' % (evaluation[0]))
print('Accuracy in Test set:  %.02f' % (evaluation[1] * 100))


# Question: how would you interpret this, especially compared to the performance in the `training set`?

# Question: What else can we do to evaluate our `ANN`? 

# How about checking the `confusion matrix`? Granted, given that we only have `2 classes` it might not be super useful. Nonetheless we might still get a bit more information on the performance of our `ANN`.

# #### Confusion matrix
# 
# For this we actually need to compute the `confusion matrix`, which we can easily do via `scikit-learn`. What we need for that are the `true` and `predicted labels`:

# In[52]:


y_true = y_test * 1
y_true


# In[53]:


y_pred = np.argmax(model.predict(X_test), axis=1)
y_pred


# Nice! Let's compute the `confusion matrix` and directly pack it into a `pandas DataFrame` for easy inspection, handling and plotting: 

# In[54]:


from sklearn.metrics import confusion_matrix
import pandas as pd

class_labels = ['closed', 'open']
cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=class_labels, columns=class_labels)

cm


# We can also plot it:

# In[55]:


import plotly.figure_factory as ff

# change each element of z to type string for annotations
z_text = [[str(y) for y in x] for x in cm.to_numpy()]

# set up figure 
fig = ff.create_annotated_heatmap(cm.to_numpy(), x=class_labels, y=class_labels, annotation_text=z_text, colorscale='Magma')

# add title
fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                  #xaxis = dict(title='x'),
                  #yaxis = dict(title='x')
                 )

# add custom xaxis title
fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=0.5,
                        y=-0.15,
                        showarrow=False,
                        text="Predicted label",
                        xref="paper",
                        yref="paper"))

# adjust margins to make room for yaxis title
fig.update_layout(margin=dict(t=50, l=200))

# add custom yaxis title
fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=-0.15,
                        y=0.5,
                        showarrow=False,
                        text="True label",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))

# add colorbar
fig['data'][0]['showscale'] = True

#fig.show()

plot(fig, filename = 'cm_eyes.html')
display(HTML('cm_eyes.html'))


# Question: how do you interpret this with regard to the performance of our `ANN`?

# #### `Layer representations`
# 
# Another thing we could do is to check what our `ANN` `learned`, that is the `representations` it computed within each `layer`. Yes, we finally made it: we will look at `learned representations`! (Remember those `latent variables` ?) We can define a short `function` that will help us with that: 

# In[56]:


from tensorflow.keras import backend as K

import matplotlib.pyplot as plt

# Specify a function that visualized the layers
def show_activation(layer_name):
    
    layer_output = layer_dict[layer_name].output

    fn = K.function([model.input], [layer_output])
    
    inp = X_train[0:1]
    
    this_hidden = fn([inp])[0]
    
    # plot the activations, 8 filters per row
    plt.figure(figsize=(16,8))
    nFilters = this_hidden.shape[-1]
    nColumn = 8 if nFilters >= 8 else nFilters
    for i in range(nFilters):
        plt.subplot(nFilters / nColumn, nColumn, i+1)
        plt.imshow(this_hidden[0,:,:,i], cmap='magma', interpolation='nearest')
        plt.axis('off')
    
    return


# Additionally, we will need the `names` of our `layers`: 

# In[57]:


layer_dict = dict([(layer.name, layer) for layer in model.layers])


# Now we can simply call it, providing the name of the `layer` which `representation` we want to check. For example, the `input layer`:

# In[58]:


show_activation('conv2d_3')


# How about the next `convolutional layer`? Remember: it should get more abstract! 

# In[59]:


show_activation('conv2d_4')


# And finally, the last `convolutional layer`:

# In[60]:


show_activation('conv2d_5')


# Fancy, eh? How would describe this and do you think this helpful to understand what our `ANN` does?

# Are there any more options that come to your mind how we can further evaluate our (pre-) trained `ANN`?

# How about going back to the basic concepts and thinking about `generalization`? We know how our `ANN` performs in the hold-out `test set` of our `dataset` but what about completely different `data`? `Data` that varies more or less prominently in several aspects (think about what these aspects could be)? Remember: we're interested in `invariant representations` and thus, if our `ANN` really `learned` something `generalizable`. In turn, we should be able to feed it a different `dataset` with diverging specifics and yet obtain sensible outcomes, i.e. a good `performance`. 

# #### `Generalization`
# 
# That being said: let's bring back an old friend, our example `dataset` from the "classic" `machine learning` part. It's also a `resting state fMRI dataset`, but different in terms of `participants`, `data acquisition sequence`, etc. . Previously, we worked with `vectorized connectivity matrices` but now we need the `fMRI volumes/images`. Thus, we need to download them first:

# In[61]:


import urllib.request

url = 'https://www.dropbox.com/s/73xlwtcochbytpv/dataset_ML_eval.nii.gz?dl=1'
urllib.request.urlretrieve(url, 'dataest_ML_eval.nii.gz')


# As usual, let's load them and have a brief look:

# In[62]:


dataset_eval = nb.load('dataest_ML_eval.nii.gz')
dataset_eval.shape


# In[63]:


dataset_eval.orthoview()


# To once more accommodate the restricted time and resource situation, we didn't include the entire `dataset`, but only the `middle volume` of each `participants`' `fMRI images`.

# As mentioned several times before: one of the crucial parts of re-using existing `ANN`s on new `data` is to bring the new `data` into the format that is expected by the `ANN`, i.e. the `input layer`.  Thus, we need to prepare the `data` a bit... (Please note: this is something you'll have to do as well when re-using existing `ANN`s with your data and therefore always make sure to consult the respective documentation and/or respective publication.) 

# Going back to the beginning of this section, we remember that our `input layer` expects the `data` in the form of `samples, x, y, z` and that we submitted only a couple of `slices` of the `image` after we `rotated` it. Let's just re-use the respective code (Please note: Always be careful with that and check things more than once, copy+paste can be _very_ dangerous!) 

# In[64]:


from nilearn.image import resample_to_img

# we re-load the initial dataset here
data = nb.load('dataest_ML.nii.gz')

# we resample the images our new dataset to those of the initial dataset
dataset_eval_resmp = resample_to_img(dataset_eval, data)

# we rotate our new dataset
dataset_eval_affine = nb.Nifti1Image(dataset_eval_resmp.get_fdata(), new_affine)
dataset_eval_rot = resample_img(dataset_eval_affine, data.affine, interpolation='continuous')

# we get the slices of our new dataset
dataset_eval_slab = dataset_eval_rot.slicer[..., 12:15, :]

# we change the dimensions of our new dataset's slab
data_eval = np.rollaxis(dataset_eval_slab.get_fdata(), 3, 0)
data_eval.shape


# That looks about right, but we can also make sure that the dimensions of our initial `dataset` and our new `dataset` fit:

# In[65]:


data.shape[1:] 


# In[66]:


data_eval.shape[1:]


# Great, now we can already put our `ANN` back to work and let it predict the new `dataset`:

# In[67]:


dataset_eval_y_pred = np.argmax(model.predict(X_test), axis=1)
dataset_eval_y_pred


# Ok, worked like a charm! However, how can we evaluate the `ANN`'s performance on the new `dataset`? 

# Ha, gotcha: We simply can not! The new `dataset` we were investigating doesn't have any `labels` useful for the task our `ANN` was `built` and `trained` to do, that is the recognition and distinction of `eyes open` vs. `eyes closed`. It will of course still work, i.e. `predict`, because it doesn't know that but we just have no option to evaluate its `performance`.
# 
# <img align="center" src="https://c.tenor.com/W3ThqxOhD-cAAAAC/gotcha-fooled-you.gif" alt="logo" title="Github" width="300" height="480" />
# <sub><sup><sub><sup><sup>https://c.tenor.com/W3ThqxOhD-cAAAAC/gotcha-fooled-you.gif
# </sup></sup></sub></sup></sub>
# 
# 
# 

# However....we do have `labels` for this new/old `dataset`, namely for the `task(s)` we tried during the "classic" `machine learning` part: `predicting` a `participant`'s `age` or `age group`. So what we theoretically could do is to assume that the `representations` our `ANN` `learned` will also help it to achieve this new `task`. Does this ring a bell?  

# #### Transfer learning
# 
# That's right folks: it's about `transfer learning`. The idea would be that the `representations` `learned` by our `ANN` to recognize and distinguish `eyes open` vs. `eyes closed` will also help it to recognize and distinguish `age groups`. Please note: we're leaving the reasonableness behind here as we want to showcase `transfer learning` but can you think of potentially more feasible `transfer learning problems`?

# Thinking back to the [previous section](), we talked about how `transfer learning` works in theory: we "simply" `re-train` the `output layer` or several `layers` for the new `task`. In practice this means we will remove the `learned weights` of these `layers` and `"freeze"` the other ones, thus only the first will be updated based on the new `task` and the corresponding `loss function` and `optimization`.

# Question: why `re-train` at all and not just simply providing the new `data` and respective `labels`?

# Ok, but how do we do this `transfer learning`? At first, we will get the `labels` in order to define our `training` and `test set`.

# In[68]:


import pandas as pd
information = pd.read_csv('participants.csv')
information.head(n=5)


# In[69]:


Y_cat = information['Child_Adult']
Y_cat.describe()


# It's been a while, so let's plot them:

# In[70]:


import plotly.express as px
from IPython.core.display import display, HTML
from plotly.offline import init_notebook_mode, plot

fig = px.histogram(Y_cat, marginal='box', template='plotly_white')

fig.update_layout(showlegend=False, width=800, height=800)
init_notebook_mode(connected=True)

#fig.show()

plot(fig, filename = 'labels_dl_eval.html')
display(HTML('labels_dl_eval.html'))


# Alright, creating `training` and `test datasets` is old news for you by now: 

# In[71]:


age_class = information.loc[Y_cat.index,'Child_Adult']

Y_cat = Y_cat =='adult'

X_train, X_test, y_train, y_test = train_test_split(data_eval, Y_cat, random_state=0, shuffle=True, stratify=age_class)

print('Shapes of X:', X_train.shape, X_test.shape)
print('Shapes of y:', y_train.shape, y_test.shape)


# Now we will `save` our `ANN` to have its `architecture` and especially `weights` on file.

# In[72]:


model.save('ANN_eyes')


# We could also just have saved the `weights` like so:

# In[73]:


model.save_weights('ANN_eyes_weights')


# Depending on the `model` and the way folks provided it, you might encounter several of these and other options when you want to use `pre-trained models`. 

# Now it's time to define our new `ANN` via basically loading the one we just saved into a new instance:

# In[74]:


from tensorflow.keras.models import load_model

model_age = load_model('ANN_eyes')


# We can easily make sure the `ANN` looks as expected:

# In[75]:


model_age.summary()


# Check, that looks as it should. What follows is the central part: we tell the `ANN` that only its last, i.e. `output layer`, will be `trainable`, i.e. capable of updating its `weights` with the others remaining the same: 

# In[76]:


model_age.load_weights('ANN_eyes_weights')

for layer in model_age.layers[:-1]:
    layer.trainable = False


# In[77]:


model_age.summary()


# The rest of steps are identical to the first time we utilized our `ANN`: `building` via the `.compile()` function, `training` via the `.fit()` function and then inspecting its performance. We will keep the same `metric`, `loss function` and `optimizer` as before:

# In[ ]:


model_age.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', 
              metrics=['accuracy'])


# The same holds true for number of `epochs`, `batch size` and the `validation split`, logging everything to use `tensorboard` again after the `training`:

# In[ ]:


logdir = os.path.join("logs_age", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


# In[82]:


get_ipython().run_line_magic('time', 'fit_age = model_age.fit(X_train, y_train, epochs=125, batch_size=32, validation_split=0.2, callbacks=[tensorboard_callback])')


# Significantly faster than `training` the entire `ANN`! We will use the same approach as during the evaluation of our `initial ANN` to evaluate how well `transfer learning` worked: plotting `metric` and `loss` for the `training` and `test set` across `epochs`, checking things via `tensorboard` and computing a `confusion matrix`:

# In[83]:


import plotly.graph_objects as go
import numpy as np

epoch = np.arange(nEpochs) + 1

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=epoch, y=fit_age.history['accuracy'],
                    mode='lines+markers',
                    name='training set'))
fig.add_trace(go.Scatter(x=epoch, y=fit_age.history['val_accuracy'],
                    mode='lines+markers',
                    name='validation set'))

fig.update_layout(title="Accuracy in training and validation set",
                  template='plotly_white')

fig.update_xaxes(title_text='Epoch')
fig.update_yaxes(title_text='Accuracy')

#fig.show()

plot(fig, filename = 'acc_age.html')
display(HTML('acc_age.html'))


# Question: how would you interpret this?

# In[84]:


import plotly.graph_objects as go
import numpy as np

epoch = np.arange(nEpochs) + 1

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=epoch, y=fit_age.history['loss'],
                    mode='lines+markers',
                    name='training set'))
fig.add_trace(go.Scatter(x=epoch, y=fit_age.history['val_loss'],
                    mode='lines+markers',
                    name='validation set'))

fig.update_layout(title="Loss in training and validation set",
                  template='plotly_white')

fig.update_xaxes(title_text='Epoch')
fig.update_yaxes(title_text='Loss')

#fig.show()

plot(fig, filename = 'loss_age.html')
display(HTML('loss_age.html'))


# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir logs_age')


# How about the `performance` on the `test set`?

# In[85]:


evaluation = model_age.evaluate(X_test, y_test)
print('Loss in Test set:      %.02f' % (evaluation[0]))
print('Accuracy in Test set:  %.02f' % (evaluation[1] * 100))


# The `confusion matrix` might be interesting...

# In[86]:


y_true = y_test * 1

y_pred = np.argmax(model.predict(X_test), axis=1)

class_labels = ['child', 'adult']
cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=class_labels, columns=class_labels)

z_text = [[str(y) for y in x] for x in cm.to_numpy()]

fig = ff.create_annotated_heatmap(cm.to_numpy(), x=class_labels, y=class_labels, annotation_text=z_text, colorscale='Magma')

fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                  #xaxis = dict(title='x'),
                  #yaxis = dict(title='x')
                 )

fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=0.5,
                        y=-0.15,
                        showarrow=False,
                        text="Predicted label",
                        xref="paper",
                        yref="paper"))

fig.update_layout(margin=dict(t=50, l=200))

fig.add_annotation(dict(font=dict(color="black",size=14),
                        x=-0.15,
                        y=0.5,
                        showarrow=False,
                        text="True label",
                        textangle=-90,
                        xref="paper",
                        yref="paper"))

fig['data'][0]['showscale'] = True

#fig.show()

plot(fig, filename = 'cm_age.html')
display(HTML('cm_age.html'))


# Any idea what's going on here?

# Did `transfer learning` work here? Think about that we wanted to go from `eyes open` vs. `eyes closed` to `child` vs. `adult` and only `re-training` our `output-layer`: `learning` `representations` to recognize and distinguish if participants had their eyes open or closed is presumably very different than `learning` `representations` to recognize and distinguish the age of participants. Additionally, we could `"un-freeze"` some of the other, lower `layers` and `re-train` them as well or even try a completely new `ANN` architecture and `data input` (e.g. more `slices`). The first would be referred to as `fine-tuning` where the `learned weights` are used as `initialization` but the entire `ANN` or `higher layers` will be `re-trained`.
