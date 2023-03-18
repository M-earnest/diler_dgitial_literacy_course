#!/usr/bin/env python
# coding: utf-8

# # Excercise: Introduction to Git and GitHub
# 
# [Michael Ernst](https://github.com/M-earnest)  
# Phd student - [Fiebach Lab](http://www.fiebachlab.org/), [Neurocognitive Psychology](https://www.psychologie.uni-frankfurt.de/49868684/Abteilungen) at [Goethe-University Frankfurt](https://www.goethe-university-frankfurt.de/en?locale=en)

# ### Before we get started...
# <br>
# 
# - most of what you’ll see within this lecture was prepared by Kendra Oudyk and further adapted by Peer Herholz 
# - based on the Software Carpentries "[Version control with Git](https://swcarpentry.github.io/git-novice/)" under CC-BY 4.0
# 
# [Peer Herholz (he/him)](https://peerherholz.github.io/)  
# Research affiliate - [NeuroDataScience lab](https://neurodatascience.github.io/) at [MNI](https://www.mcgill.ca/neuro/)/[MIT](https://www.mit.edu/)  
# Member - [BIDS](https://bids-specification.readthedocs.io/en/stable/), [ReproNim](https://www.repronim.org/), [Brainhack](https://brainhack.org/), [Neuromod](https://www.cneuromod.ca/), [OHBM SEA-SIG](https://ohbm-environment.org/), [UNIQUE](https://sites.google.com/view/unique-neuro-ai)  
# 
# <img align="left" src="https://raw.githubusercontent.com/G0RELLA/gorella_mwn/master/lecture/static/Twitter%20social%20icons%20-%20circle%20-%20blue.png" alt="logo" title="Twitter" width="32" height="20" /> <img align="left" src="https://raw.githubusercontent.com/G0RELLA/gorella_mwn/master/lecture/static/GitHub-Mark-120px-plus.png" alt="logo" title="Github" width="30" height="20" />   &nbsp;&nbsp;@peerherholz 

# ## Roadmap
# 
# - **Goals**
# - Setup
# - Why use git & GitHub?
# - Where does git store information?
# - How do I record changes in git?
# - How do I share my changes on the web?
# - How do I contribute to an existing project?
# - Goals

# ## Goals
# 1. Explain why git/GitHub are useful
# 2. Track and share your work using git/GitHub (git level 1: commit push)
# 3. Contribute to a project using git/GitHub (git level 2: branches PRs)

# ## On learning git & GitHub
# ![](static/doing_and_understanding_cleaner.png)

# ## Roadmap
# 
# - Goals
# - **Setup**
# - Why use git & GitHub?
# - Where does git store information?
# - How do I record changes in git?
# - How do I share my changes on the web?
# - How do I contribute to an existing project?
# - Goals

# ## Import side note:
# 
# **throughout this lecture/practice you will see a lot of things in <>, within each instance please replace <> with your own info**

# e.g.,
# 
# `github.com/<your_username>`
# 
# becomes
# 
# `github.com/peerherholz`

# ## Setup
# #### To follow on your machine, you'll need
# 1. Bash
# 2. Git
# 3. Text editor
# 4. GitHub account
# 
# _For an explanation on how to get those plesase check back with the [setup]() section_

# ### Check if you're ready
# 1. Can you open a text editor? (e.g., Linux: gedit, nano. macOS: textedit. Windows: notepad)
# 2. Can you go to your GitHub account?
# 3. When you open a Bash shell and type `git --version`, does it output the version number? (**macOS / Linux**: you might need to run this: `conda install -c anaconda git`)
# 

# ### Configure git (if you haven't already)
# 
# As git doesn't work streaight out of the box, we'll have to first provide our credentials. Just copy paste the following lines into a bash shell (remember how to open those?).
# 
# ```
# git config --global user.name "<Vlad Dracula>"
# git config --global user.email "<vlad@tran.sylvan.ia>"
# ```
# *use the email you used for your GitHub account*  👆
# 
# #### macOS / Linux
# ```
# git config --global core.autocrlf input
# ```
# 
# #### Windows
# ```
# git config --global core.autocrlf true
# 
# ```

# ## Roadmap
# 
# - Goals
# - Setup
# - **Why use git & GitHub?**
# - Where does git store information?
# - How do I record changes in git?
# - How do I share my changes on the web?
# - How do I contribute to an existing project?
# - Goals

# <img align="top" src="https://uidaholib.github.io/get-git/images/phd101212s.gif" alt="git-phd" title="git-phd" width="550" height="500" />
# 
# “Piled Higher and Deeper” by Jorge Cham, http://www.phdcomics.com

# ### Why use git & GitHub?
# 
# **Automated version control**

# ### Record versions by tracking *changes*
# It's like having an unlimited "undo" button
# ![](https://swcarpentry.github.io/git-novice/fig/play-changes.svg)

# ### Make independent changes
# ![](https://swcarpentry.github.io/git-novice/fig/versions.svg)

# ### And incorporate the changes
# 
# ![](https://swcarpentry.github.io/git-novice/fig/merge.svg)
# 
# https://swcarpentry.github.io/git-novice/

# ## Roadmap
# 
# - Goals
# - Setup
# - Why use git & GitHub?
# - **Where does git store information?**
# - How do I record changes in git?
# - How do I share my changes on the web?
# - How do I contribute to an existing project?
# - Goals

# Open your Bash shell (where you typed `git --version` at the beginning)
# 
# Create a directory (remember Windows' slashes are the other way) bing the following lines of bash code into your shell.

# In[ ]:


cd ~/Desktop
mkdir desserts
cd desserts


# To check what's in our directory, we can use the bash command `ls`

# In[ ]:


ls -a


# To create create a git repository, we can simply run the following command in our shell

# In[ ]:


git init


# What's in our directory now?

# In[ ]:


ls -a


# **The `.git` subdirectory is where git stores all the info it needs to do version control**

# ## Roadmap
# 
# - Goals
# - Setup
# - Why use git & GitHub?
# - Where does git store information?
# - **How do I record changes in git?**
# - How do I share my changes on the web?
# - How do I contribute to an existing project?
# - Goals

# ![](static/workflow/w0_init.png)

# ![](static/workflow/w1_local.png)

# ### `git add`
# ![](https://i.gifer.com/YKCS.gif)
# ### `git commit`
# ![](https://www.nydailynews.com/resizer/nJ3qGqkV_0Z6WzIGAWktQ0pKlIE=/415x229/top/arc-anglerfish-arc2-prod-tronc.s3.amazonaws.com/public/JOYD6SAJXDW4JQJSKWAZIY266Y.jpg)

# Let's make a change!
# First, open a new file
# ```
# <text editor> desserts.md
# ```

# Write this in the file:
# 
# > pie\
# > ice cream\
# > cookies
# 
# Save and exit

# Reminder to let me know if you need me to slow down
# ![](https://goldstarteachers.com/wp-content/uploads/2018/09/Confused-GIF.gif)

# Let's check the status of our repo

# In[ ]:


git status


# Is this file being tracked by git?
# ![](figures/zoom_icons/poll.png)
# 
# <font color='grey'>(hint: look at what your terminal says)</font>

# How can we include this file in what will be committed?
# ![](figures/zoom_icons/chat_mic.png)

# Let's stage the change

# In[ ]:


git add desserts.md


# Let's check the status of our `repo`

# In[ ]:


git status


# Let's commit the change

# In[ ]:


git commit -m "list my favorite desserts"


# Let's check the status of our `repo`

# In[ ]:


git status


# **I change my mind...**
# 
# **cookies are better than ice cream**

# ```
# $ <text editor> desserts.md
# ```
# 
# > pie\
# > cookies\
# > ice cream
# 
# Save and exit

# Let's \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_
# ```
# git diff
# ```
# 
# How could we figure out what this command does?
# ![](figures/zoom_icons/chat_mic.png)

# Let's stage and commit the change
# ```
# git ____ desserts.md
# git ____ -m "switch cookies and ice cream"
# ```
# ![](figures/zoom_icons/chat_mic.png)

# ## Check your understanding

# ### What does git track?
# ![](https://swcarpentry.github.io/git-novice/fig/play-changes.svg)

# ### Does git track changes to each letter?

# ### How do I get Git to track a change?

# Put these in order:
# 
# a) `git commit -m "<this is what I did>"`\
# b) make the change\
# c) `git add <file>`
# 
# ![](figures/zoom_icons/poll.png)

# ## Roadmap
# 
# - Goals
# - Setup
# - Why use git & GitHub?
# - Where does git store information?
# - How do I record changes in git?
# - **How do I share my changes on the web?**
# - How do I contribute to an existing project?
# - Goals

# ![](static/workflow/w2_inspect.png)

# ![](static/workflow/w3_remote.png)

# ## Create a remote repo

# - Go to [github.com](https://github.com/)

# - Beside **Repositories**, click **New**

# - Enter your repo name
# - Choose to make your repo Public or Private
# - Don't check any boxes
# - Click **Create repository**

# ## Link it to your local repo

# Tell git the URL of your `remote repo` and name it 'origin'

# In[ ]:


git remote add origin https://github.com/<yourusername>/desserts.git


# Set the name of your principle branch to main (if it's not already)

# In[ ]:


git branch -M main


# Push your changes to `GitHub`

# In[ ]:


git push -u origin main


# Refresh your GitHub repo

# ## Roadmap
# 
# - Goals
# - Setup
# - Why use git & GitHub?
# - Where does git store information?
# - How do I record changes in git?
# - How do I share my changes on the web?
# - **How do I contribute to an existing project?**
# - Goals

# ![](static/workflow/w4_update.png)

# ![](static/workflow/w5_upstream.png)

# ## Branches

# ![](static/branches_features.png)

# ![](static/branches_collab.png)

# #### I want to contribute!

# ##### Contributing task 1: Get everyone's favorite desserts!
# 
# 

# ## Roadmap
# 
# - Goals
# - Setup
# - Why use git & GitHub?
# - Where does git store information?
# - How do I record changes in git?
# - How do I share my changes on the web?
# - How do I contribute to an existing project?
# - **Goals**

# ### Did we meet our goals?

# #### 1. Explain why git & GitHub are useful

# ... to a new grad student
# ![](figures/zoom_icons/chat_mic.png)

# #### 2. Track and share your work using git/GitHub (git level 1: commit push)
# 

# `status` &nbsp; &nbsp;
# `add` &nbsp; &nbsp;
# `init` &nbsp; &nbsp;
# `commit` &nbsp; &nbsp;
# `diff` &nbsp; &nbsp;
# `push` &nbsp; &nbsp;
# 
# **Basic workflow for tracking a change and putting it on GitHub**
# - make a change
# - stage the change: `git ____ <filename>`
# - commit the change: `git ____ -m "<commit message>"`
# - put the change on GitHub: `git ____ origin main`

# **See what's happening with git**
# - show the working tree status: `git ____`
# - show how the file changed: `git ____`

# #### 3. Contribute to a project using git/GitHub (git level 2: branches PRs)

# ##### Contributing task 2: Correct spelling mistakes in desserts lists
# 
# ![](figures/issues.png)

# ![](https://media1.tenor.com/images/ae1fd92f4ed82fba165d777e4a05c9de/tenor.gif?itemid=14220287)

# ## There's so much more!

# ### Git buffet
# ![](static/git_github_buffet.png)

# ### Git is hard
# Here are some tips

# ### Sit down and go through a tutorial
# ![](static/swc_git_website.png)
# 
# ![](static/swc_coverage.png)

# ### Don't expect to remember everything
# ![](static/google_stuff.png)

# ### Keep common commands on a sticky note
# ![](static/sticky_note_on_laptop.jpg)

# #### To learn it, you need to *commit* to doing it
# ![](static/doing_and_understanding_cleaner.png)

# ## Quick feedback
# 
# ##### How much of this tutorial could you follow?
# - 100 %
# - 75 %
# - 50 %
# - 25 %
# - 0 %
# 
# ##### Where are there any major hurdles?

# ## The End
# 
# 
# Software Carpentry's tutorial: https://swcarpentry.github.io/git-novice/
# 
# ![](https://media.riffsy.com/images/f9fd6fdf307421f068d82cd050eae236/tenor.gif)
# 
