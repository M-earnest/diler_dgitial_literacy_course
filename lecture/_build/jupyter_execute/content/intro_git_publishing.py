#!/usr/bin/env python
# coding: utf-8

# # Git, GitHub and hosting websites using GitHub Pages

# ToDo: Missing pictures, alt-text and quick tutorial, formatting, incorporate some of https://peerherholz.github.io/Cog_Com_Neuro_ML_DL/introduction/notebooks/neurodata/version_control.html
# 
# 
# To publish your course/website we will be relying on Github, a web-based platform that provides hosting and storage services for software and data. 
# 
# Github, or the underlying version control system "Git", is basically used in every large software project that you can think of. A Github server may be setup on a local machine or network, if the access to the hosted software or data should be restricted, but we'll be using the public Github server, i.e. the [web-based version of Github](https://github.com) hosted. 
# In essence, this allows us to openly share and for other people to make copies or download our projects, e.g. for collaborative development or issue tracking, it further allows for feedback on our courses or the reporting of possible errors people may have encountered.
# 
# This project is build on the idea that the knowledge that we're trying to share and that the courses build on this framework should be as accessible as possible, therefore we opted to host this [course](), as well as our [course template]() via Github. This allows you and others to simply download the contained materials and adapt them to make your own projects. Further Github allows us to host websites like this one via the "Github pages" service free of any charges.
# 
# The following lesson will contain information on how to setup you public Github account, how to setup a project on github contained in github "repository", how to use the version control software "Git" to manage your projects and "push" your local course materials to your online repository and how to host your course website using Github pages.
# 

# [Michael Ernst](https://github.com/M-earnest)  
# Phd student - [Fiebach Lab](http://www.fiebachlab.org/), [Neurocognitive Psychology](https://www.psychologie.uni-frankfurt.de/49868684/Abteilungen) at [Goethe-University Frankfurt](https://www.goethe-university-frankfurt.de/en?locale=en)
# 
# ## Before we get started...
# <br>

# ## Goals
# 
# This chapter aims to include the following goals. Do check back here later and you feel like the lesson addressed the stated goals appropriately!
# 
# 1. Understand the benefits of using Github, including why a version 
# 2. Learn how to setup your Github account
# 3. Learn how to setup and connect local and online (remote) repostiory
# 5. Learn how to record record changes via git
# 6. learn how to submit my changes to an online repository
# 7. learn how to contribute to an existing project
# 8. Learn how to host a website using Github pages
# 
# 
# #### Roadmap
# 
# - **What's Github and why should you use it? (skip this if you're already familiar with the concept**
# - Setting up a Github account
# - Start a project/setup a public repository
# - Setup your local maching for working with Git/Github
# - Getting started with Gtihub/Github Workflow
# - Working with the Course template
# - Hosting a website using Github pages
# - Quick tutorial
# 
# 

# ### What's a "Github" and why should we use it?
# 
# 
# As already mentioned, Github is a web-based plattform for store, organize, and managing projects.
# 
# Content is mainly organized in repositories (or "repos" for short), these repos can be viewed as being equivalent to a normal directory, that you have on your local machine for e.g. storing data, pictures.
# They may contain a variety of file-types and nested directories, and will be used as a place to store and organize your project files, including code, assets and documentation
# 
# Here you see the repository for the course template that you'll be working with. Don't worry if that's still confusing to you, learning how to work with Github is best done using a "learning-bydoing" approach.
# 
# ----------------- Insert picture ---------------------------
# 
# 
# 
# 
# #### Version control and collaborative work
# 
# I'm sure we've all lost important document, code or progress while working in academia. If you've been doing programming work, you might also have encountered that fixing a bug may lead to othe substantial bugs down the line. 
# So people generally tend to come up with their on solution for "project managment", e.g. by maintaining multiple files, e.g. project_working_copy_1.txt, project_notes.txt,  project_final_draft_3 or at times by making a mess of our documents by storing cut or to be implemented content in it's own sub-section at the end of project_draft_4.txt.
# 
# Github alleviates that problem by making use of "Git" a distributed version control system, which logs each time a change is made to a file. Via the commit system we can not only maintain different versions of the same file on e.g. your local system and the connected public repository, but also revert changes that had negative consequences or recover information lost on your system.
# 
# Git further makes it easy to work with multiple users on the same project, as each user can maintain their local versions and submit their changes to the public directory, that can then be reviewed by the team on quality and compatibility and subsequently be incorporated into the online project.
# 
# 

# #### Roadmap
# 
# - What's Github and why should you use it? (skip this if you're already familiar with the concept
# - **Setting up a Github account**
# - Start a project/setup a public repository
# - Setup your local maching for working with Git/Github
# - Getting started with Gtihub/Github Workflow
# - Working with the Course template
# - Hosting a website using Github pages
# - Quick tutorial

# ### Setting up a Github account
# 
# First things, first let's setup a GitHub Account. 
# 
# **1. Go to the GitHub website**: Open a web browser and navigate to the [GitHub website](https://github.com/).
# 
# **2. Click on the "Sign up" button**: On the home page, click on the "Sign up" button located in the upper-right corner.
# 
# **3. Fill out the registration form**: Fill out the required information and choose a username. As your username will be included in the link to the website we're going to build and will be publicly displayed choose carefully!
# 
# **4. Verify your email address**: Check your email inbox and click on the verification link sent by GitHub.
# 
# **5. Customize your profile**: Fill out your profile information and add a profile picture. This will help others to find and connect with you on GitHub.
# 
# </br>
# 
# 
# **And that's it!**
# You can now use your new GitHub account store your code online or collaborate and share your projects with the world!
# 

# #### Roadmap
# 
# - What's Github and why should you use it? (skip this if you're already familiar with the concept
# - Setting up a Github account
# - **Start a project/setup a public repository**
# - Setup your local maching for working with Git/Github
# - Getting started with Gtihub/Github Workflow
# - Working with the Course template
# - Hosting a website using Github pages
# - Quick tutorial

# ### Start a project/setup a public repository
# 
# Let's put that new account to use, by creating a new online repository, often referred to as a "remote repo"!
# 
# 
# 1. Open Github in your browser
# 
# 2. Click on the `+` sign in the top right corner and click "New repository"
# 
# 3. Fill out the repository details: Give your repository a name, description, and check the box next to "public" to make sure others can find your directory. 
# 
# This could look something like this:
# 
# 
# <img src="https://github.com/felixkoerber/jb/blob/main/static/new_repo_example.png?raw=true" alt="depicting an example of a new repository" class="bg-primary" width="500px">
# 
# 
# 4. Check the box "Add a README file", this will initiate your repository with a file that can later be used to display basic information to others viewing your repo.
# 
# 5. Choose a license! You can start out with "None" as this repo is just for testing, but if you plan to use Github for your projects it's imperative to include one to prevent misuse. Find more info [here]().
# 
# 
# 6. Click on the "Create repository" button to create your new repository.
# 
# 7. Now you can add files to your repository by clicking the "Add file" button by either uploading them or creating a new file.
# 
# Congratulations! You are now the proud owner of a Github repository. In the following parts we'll be diving a bit deeper into the specifics of how a standard Github-Workflow might look like, how to connect your local system to your online repos ("cloning"). 
# 

# #### Roadmap
# 
# - What's Github and why should you use it? (skip this if you're already familiar with the concept
# - Setting up a Github account
# - Start a project/setup a public repository
# - **Setup your local maching for working with Git/Github**
# - Getting started with Gtihub/Github Workflow
# - Working with the Course template
# - Hosting a website using Github pages
# - Quick tutorial

# 
# ## Setup
# 
# To use Git on your local machine to log any changes you make to your project and submit your changes to the internet, you'll be needing the following:
# 
# - a GitHub account
# - [Git](https://git-scm.com/downloads)
# - [a Text editor (e.g. VScode)](https://code.visualstudio.com/)
# - Bash
# 
# We've already discussed how to setup a Github account and create an online repo. How to get the rest of the listed software is explained in detail in our complete [Setup section](link to setup). 
# 
# You may **skip the installation of Bash and instead install the [Gitkraken client](https://www.gitkraken.com/)**, a software package icnluding a Graphical-user-interface to manage all your git/Github projects. If you're not too familiar with coding or simply prefer to have a visual overview over the exact changes you've made, your version history etc. this might just be the better choice for you.
# 
# Simplye downlowad and install the [Gitkraken client](https://help.gitkraken.com/gitkraken-client/how-to-install/) and [conncet it to your online Github profile](https://www.youtube.com/watch?v=5nhNfMcczlQ).
# 
# **Check if you're ready**
# 
# * Can you open a text editor? (e.g., Linux: gedit, nano. macOS: textedit. Windows: notepad)
# * Can you go to your GitHub account?
#     
# * A.) When you open a Bash shell and type git --version, does it output the version number? (macOS / Linux: you might need to run this: conda install -c anaconda git)
#     
#     or 
#     
# * B.) Open Gitkraken and create a new repository (called intiating a new repository, under "Start a local project"
# 
# 

# #### Roadmap
# 
# - What's Github and why should you use it? (skip this if you're already familiar with the concept
# - Setting up a Github account
# - Start a project/setup a public repository
# - Setup your local maching for working with Git/Github
# - **Getting started with Gtihub/Github Workflow**
# - Working with the Course template
# - Hosting a website using Github pages
# - Quick tutorial

# ## Getting started
# 
# Now getting started with Github may feel overwhelming, but for this course we'll be learning just the bare minimum.
# 
# If you're confused about the language used or want to understand things in greater detail, there are great ressources ou there for you to explore:
# 
# - [Git definitions](https://www.gitkraken.com/learn/git/definitions)
# 
# - [Git tutorials](https://www.gitkraken.com/learn/git/tutorials)
# 
# - [Learning Git with Gitkraken](https://help.gitkraken.com/gitkraken-client/gitkraken-client-home/)
# 
# </br>
# </br>
# 
# 
# ### Git operations
# 
# 
# Git can be quite complex, but **we'll only be using the following operations**, for in-depth explanations in video form click the contaned links:
# 
# [**Forking**](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks): Creating a copy of someones repository, that you want to work with. Forks let you make changes to a project without affecting the original repository. I.e. you'll create an online copy of anothers repository under your github account.
# 
# 
# [**Cloning**](https://www.gitkraken.com/learn/git/tutorials/what-is-git-remote): Meaning to create a local copy of an online repository (called "remote"s on your system.
# 
# 
# [**Branching**](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches): Meaning to create a copy of a project either onliny or locally, so that you have an isolated copy where you can work without affecting other branches in a repository. Each repository has one default branch, usually called "main" branch, and can have multiple other branches.
# 
# [**Comitting**](https://www.gitkraken.com/learn/git/tutorials/what-is-git-commit): A commit is a "snapshot" of your repository at one specific point in time. If you make changes to a repository, you'll have to follow up with "comitting" the changes to save file changes to your Git repository. This does not mean that your changes will be lost, if you e.g. shutdown your system just that the version between a file on your system and the logged version in the github repository do not match.
# 
# 
# [**Pulling**](https://www.gitkraken.com/learn/git/tutorials/what-is-git-pull): Means simply to update you local repository with the corresponidng online (remote) repository. E.g. if you've or someone else made changes to the online version of a repository you pull or "download" these changes to your local files this way.
# 
# 
# [**Pushing**](https://www.gitkraken.com/learn/git/problems/git-push-to-remote-branch): Git push is used to upload a local repository’s content to a remote repository. Meaning that you send your locally comitted changes to your online repository.
# 
# 
# 
# </br>
# </br>
# 
# 
# ### Git Workflow
# 
# **A standard Git workflow may be looking something like this:**
# 
# 
# **1. Create a repository**: Create an online repository by clicking on the "New repository" button on the GitHub website. Give your repository a name, a description, and select it the "public" option.
# 
# **2. Clone the repository**: To start working with your repository, you'll need to "clone" it to your local computer. This will create a copy of the repository on your computer that you can work with. To clone a repository, click on the "Clone or download" button and copy the URL. 
# 
# `BASH`: Open a terminal or command prompt and input the following command: `git clone https://github.com/username/repositoryname.`Where username is your user name and repositoryname is the name of your of the repository.
# 
# `Gitkraken`: Click on `file`, `clone repo` and input where the repo should be stored on your system as well as the URl to the repo, i.e. `https://github.com/username/repositoryname.`
# 
# 
# **3. Make changes**: Now that you have a local copy of the repository, you can make changes to the code. Simply open the files in your editor and make the changes you want or create new files and folders.
# 
# **4. Commit changes**: When you're done making changes, you'll need to "commit" them to the repository. Committing a change records it in the repository's history and makes it part of the codebase. 
#     
# `BASH`:To commit a change, run the following commands in the terminal: git add ., git commit -m "Your commit message". Replace "Your commit message" with a brief description of the changes you made.
# 
# `Gitkraken`:  Click on `file`, `open repo` and select the repo containing your changes. On the left hand side you'll see the "file" overview. Under unstaged files you can review your changes. Click the grenn "stage all changes" button or right click on inidvidual files/folders and select "stage". The files will be added to the "staged" files windows. Following provide a name for your changes (commit) in the summary field below and add a short but meaningful description of your changes. Lastly, hit the green "Stage changes/changes to commit" button.
# 
#     
#    stage_menu.png
# 
# **5. Push changes**: Finally, you'll need to "push" your changes to the remote repository on GitHub. This will upload your changes to the website so that others can see them. 
# 
# `BASH`: To push your changes, run the following command: `git push origin master`.
# 
# `Gitkraken`: Hit the "Push" button on the upper center of the window.
# 
# 
# 

# #### Roadmap
# 
# - What's Github and why should you use it? (skip this if you're already familiar with the concept
# - Setting up a Github account
# - Start a project/setup a public repository
# - Setup your local maching for working with Git/Github
# - Getting started with Gtihub/Github Workflow
# - **Working with the Course template**
# - Hosting a website using Github pages
# - Quick tutorial

# ## Working with the course template
# 
# To work with our materials we have to add a minor step, as you don't want to push your changes to our onlince repository containing the coures template.
# 
# Instead we'll add the step of "forking" the course template repository to your account:
# 
# ### 1. Fork the course template:
# 
# Head over to the [course template repo](https://github.com/M-earnest/course_template_diler) and on the upper right hand side click "fork". 
# 
# 
# - fork_button.png
# 
# <img src="https://github.com/felixkoerber/jb/blob/main/static/fork_buttton.png?raw=true" alt="depicting positon an look of the fork button on a github repository" class="bg-primary" width="500px">
# 
# 
# You'll be asked to create a new fork, simply add a new repository name, provide a short description or keep the exiting one, check the box "Copy the main branch only" and click the "create fork" button.
# 
# 
# <img src="https://github.com/felixkoerber/jb/blob/main/static/create_fork.png?raw=true" alt="depicting positon an look of the fork button on a github repository" class="bg-primary" width="500px">
# 
# 
# Now we'll simply proceed as above:
# 
# ### 2. Clone the repository
# 
# To start working with your repository, you'll need to "clone" it to your local computer. This will create a copy of the repository on your computer that you can work with. To clone a repository, click on the "Clone or download" button and copy the URL you've used for your foked repo. 
# 
# `BASH`: Open a terminal or command prompt and input the following command: `git clone https://github.com/username/repositoryname.`Where username is your user name and repositoryname is the name of your of the repository.
# 
# `Gitkraken`: Click on `file`, `clone repo` and input where the repo should be stored on your system as well as the URl to the repo, i.e. `https://github.com/username/repositoryname.`
# 
# 
# 
# ### 3. Make changes
# 
# 
# 
# **The template structure**
# 
# 
# <img src="https://github.com/felixkoerber/jb/blob/main/static/repo.png?raw=true" alt="depicting the contents of the course template repository on github" class="bg-primary" width="500px">
# 
# 
# 
# Where our 
# 
# - .github/workflows folder: contains the prewritten scripts to automatically create your website everytime new content is pushed online
# 
# - lecture: contains all our content files and directories, as well as the "toc.yml" file, which defines the strutcture of the websites
# 
# - README: a short explanation of your website/course
# 
# - LICENSE: self-explanatory, stating who and how are people allowed to use or reproduce the content of this repo
# 
# - requirements.txt: contains the necessary requirements for the automatic scripts building the website to run, no need to change anything here.
# 
# 
# Now most the things that you'll be adapting are contained in the content folder "lecture", feel free to checkout the respective lessons on:
# 
# * [Content creation](link-to-experiments/experiments.html#)
# 
#     - [writing content](link-to-experiments/experiments.html#)
# 
#     - [Interactive content: Jupyter notebooks](link-to-experiments/experiments.html#)
# 
#     - [Style](link-to-experiments/experiments.html#)
# 
#     - [Embedding Media: Images, Videos and more](link-to-experiments/experiments.html#)
# 
# 
# * [Structuring content](link-to-experiments/experiments.html#)
# 
# 
# 
# 
# #### The README
# 
# To actually display the purpose and e.g. acknowledgments of your course you'll have to adapt the "README.md" file on your public repo. In the template this looks like this 
# 
# 
# and translates to this view at the bottom of your public remote repo
# 
# So simply add the name and explanation of your course, your credentials, and e.g. how others may get in contact with you to the README.md. Please do keep the included credit to our original G0RELLA template lectures.
# 
# 
# #### THE LICENSE
# 
# You may further change the contained LICENSE file to your liking, as long as you respect the stated stipulations.
# 
# -------- add license.png ----------
# 
# ```
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# ```
# 
# 
# 
# 
# ### 4. Commit changes
# 
#  When you're done making changes, you'll need to "commit" them to the repository. Committing a change records it in the repository's history and makes it part of the codebase. 
#     
# `BASH`:To commit a change, run the following commands in the terminal: git add ., git commit -m "Your commit message". Replace "Your commit message" with a brief description of the changes you made.
# 
# `Gitkraken`:  Click on `file`, `open repo` and select the repo containing your changes. On the left hand side you'll see the "file" overview. Under unstaged files you can review your changes. Click the grenn "stage all changes" button or right click on inidvidual files/folders and select "stage". The files will be added to the "staged" files windows. Following provide a name for your changes (commit) in the summary field below and add a short but meaningful description of your changes. Lastly, hit the green "Stage changes/changes to commit" button.
# 
#     
# <img src="https://github.com/felixkoerber/jb/blob/main/static/stage_menu.png?raw=true" alt="depicting the contents of the course template repository on github" class="bg-primary" width="300px">
# 
# ### 5. Push changes
# 
# Finally, you'll need to "push" your changes to your remote repository on GitHub. This will upload your changes to the website so that others can see them. 
# 
# `BASH`: To push your changes, run the following command: `git push origin master`.
# 
# `Gitkraken`: Hit the "Push" button on the upper center of the window.

# #### Roadmap
# 
# - What's Github and why should you use it? (skip this if you're already familiar with the concept
# - Setting up a Github account
# - Start a project/setup a public repository
# - Setup your local maching for working with Git/Github
# - Getting started with Gtihub/Github Workflow
# - Working with the Course template
# - **Hosting a website using Github pages**
# - Quick tutorial

# ## Hosting a website using Github pages
# 
# So to actually turn the course content that you pushed to your remote Github repo into a website we'll be using **Github Pages**.
# 
# GitHub Pages  allows users to host websites directly from their GitHub repositories, creating a website for your personal portfolio, project documentation, or in this case course content. 
# The website is generated directly from the contents of your GitHub repository and is automatically updated whenever changes are pushed to the repository. The the chapter structure of your website is dependent on the contents of your \_toc.yml, as shown in the [Structuring content](link-to-experiments/experiments.html#) chapter. 
# 
# Normally, you'd have to setup a ["GitHub-pages-actions"](https://github.com/marketplace/actions/github-pages-action) script for the website to be automatically build on push, but the course template already contains everything you need. 
# 
# 
# If we take a closer look at the `.github/workflows` folder, we'll find a file called `book.yml`. That already contains all the instructions necessary for GitHub to automatically build your book. All we need to do now is change a few of the settings of our online repo. 
# 
# 
# 
# 
# ### Setting up your website
# 
# **1.Open your repo in your Browser and click "Settings"**
# 
# - settings.png
# 
# <img src="https://github.com/felixkoerber/jb/blob/main/static/settings.png?raw=true" alt="depicting the contents of the course template repository on github" class="bg-primary" width="300px">
# 
# **2.Click on "Pages" under Code and automation**
# 
# <img src="https://github.com/felixkoerber/jb/blob/main/static/pages.png?raw=true" alt="depicting the contents of the course template repository on github" class="bg-primary" width="300px">
# 
# - panges.png
# 
# **3.Under Source select "deploy from branch"**
# 
# <img src="https://github.com/felixkoerber/jb/blob/main/static/deploy.png?raw=true" alt="depicting the contents of the course template repository on github" class="bg-primary" width="300px">
# 
# - deploy.png
# 
# **4.Under Branch: Select branch "master" and select the "/root" folder and save**
# 
# <img src="https://github.com/felixkoerber/jb/blob/main/static/select_branch.png?raw=true" alt="depicting the contents of the course template repository on github" class="bg-primary" width="300px">
# 
# - select_branch.png
# 
# 
# **5.Push a new commit to your repo (e.g. add a line to your README.md)**
# 
# 
# ### Checking your workflow
# 
# Following, if you now click on "Actions", at the top of your repo, you should now see that a workflow called "pages and deployment" is running. 
# 
# <img src="https://github.com/felixkoerber/jb/blob/main/static/actions_button.png?raw=true" alt="depicting the contents of the course template repository on github" class="bg-primary" width="300px">
# - actions-button.png.
# 
# 
# ### Getting your Link
# 
# Once that process has completed, i.e. is marked with a green checkmark, head back over to to "settings" -> "pages". At the top under "GitHub pages" you should now find a field that looks like this
# 
# <img src="https://github.com/felixkoerber/jb/blob/main/static/pages_link.png?raw=true" alt="depicting the contents of the course template repository on github" class="bg-primary" width="300px">
# 
# Clicking on the displayed link, should lead you to your newly build content site. Now you can simply copy that link and add it to your repos README.md, sop people can actually find your website.
# Done!
# 
# 
# ### Troubleshooting
# 
# 
# If for whatever reason the "pages and deployment" workflow fails, it will be marked with a red x, instead of a green 
# checkmark. 
# 
# <img src="https://github.com/felixkoerber/jb/blob/main/static/failed_workflow.png?raw=true" alt="depicting the contents of the course template repository on github" class="bg-primary" width="300px">
# - failed-workflow.png
# 
# 
# Clicking on the failed workflow, will reveal which process has failed, in this case the "build" process.
# 
# <img src="https://github.com/felixkoerber/jb/blob/main/static/failed-process.png?raw=true" alt="depicting the contents of the course template repository on github" class="bg-primary" width="300px">
# 
# Clicking on the process, will reveal a detailed log of which part of the process failed. Debugging this can be quite unintuitive, therefore you'll find common errors and solutions in the chapter [Troubleshooting](link.jpg). If the recommended solutions do not work or your error is not mention feel free to contact us, either via opening an [Issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue) on the [course template repo](https://github.com/M-earnest/course_template_diler) or e-mail and we'll try to get back to you ASAP.
# 
# 
# 

# #### Roadmap
# 
# - What's Github and why should you use it? (skip this if you're already familiar with the concept
# - Setting up a Github account
# - Start a project/setup a public repository
# - Setup your local maching for working with Git/Github
# - Getting started with Gtihub/Github Workflow
# - Working with the Course template
# - Hosting a website using Github pages
# - **Quick tutorial**

# ## Quick tutorial

# ## Additional ressources
# 
# If you want to start building a website from scratch, change the site-theme or write your own pages-action script from scratch, checkout the [official tutorial](https://docs.github.com/en/pages/quickstart). 
# 

# ## Acknowledgments
