#!/usr/bin/env python
# coding: utf-8

# ### Cloud Computing
# 
# Using & developing scalable and flexible computing systems that can be delivered as services over the internet.

# Cloud computing has revolutionized the way we access, store and process data. Gone are the days when accessing a centralized server using SSH was the norm, well outside of academia, big tech firms and the range of the german , that is. Today, cloud computing services offer a wide range of options for individuals and organizations to store, process and analyze data in a scalable and cost-effective manner. 
# 
# In this module, we will take a (somewhat deep) dive into the world of cloud computing, exploring both classic approaches such as accessing a server using SSH, as well as newer approaches such as cloud computing services suitable for research. 
# 
# In short, this chapter will provide a comprehensive overview of the cloud computing landscape and what you might encounter.

# ## Goals
# 1. 
# 
# 
# ## Roadmap
# 
# - **Goals**
# - Big Data Analytics
#     - FAIR principles
# - Machine Learning/AI
# - Cloud Computing
# 
# 
# 

# 
# 
# ### Cloud computing & SSH
# 
# SSH (Secure Shell) is a secure protocol used to remotely access and manage computer systems over an unsecured network. Or simpler it provides a user access to a remote computer or computer systems, by defining a set of rules and procedures for transmitting data between computers.
# It uses encryption protocols to protect sensitive information exchanged between devices, making it a secure way to access remote servers and devices. SSH is therefore commonly used by organizations or insitutions to access or analyse sensitive data. 
# 
# In academia or clinical insitutions, e.g. in neuroscience research, you may encounter SSH as a means to log in to a remote server and execute commands, transfer large files, or perform analysis that a regular computer could not handle or that would be incredibly time comsuming.

# </br>
# 
# If your work requires you to work with SSH, you'll be usually introduced to the local IT support, that teaches you how to access their server systems and what permissable operations you might perform. Here is what you can expect from this interaction
# 
# 
# 1. Depending on your operating system you will either be installing a SSH client on your machine (e.g.  Solar-PuTTY for Windows), on MacOS and Linux you'll instead use the terminal. Normally you will provided access to a computer designated for that purpose using macOS/Linux
# 
# 2. Next a public/private key pair for you local machine will be generated. This is used to authenticate your machine to the remote server. Think of it as a work ID that your computer gets assigned, which will be checked against a list of registered machines, when accessing the remote server. 
# 
# 3. Now open the installed SSH client or terminal on your local machine and enter the following command
#    
#    `ssh <user>@<remote_server_IP>`
#     
#     (where user is your username on the remote server and remote_server_IP the IP address or hostname of the remote server. Both will most liekly be provided by the system administrator.)
# 
# 4. You will be prompted to enter your password. Once you enter the correct password, you will be logged into the remote server over an encrypted SSH connection.
# 
# But you'll probably soon realize somethin shocking: the lack of the GUI (Graphical User Interface)!
# As SSH is a service used for managing data in the most efficient way we're usually constrained to text-based communication with the machine. The most common language used for this purpose is called BASH (Bourne Again SHell). 
# 
# BASH (Bourne-Again SHell) is a widely used command-line interface (CLI) for Unix-based operating systems. If you open up a on your [macOS terminal](https://support.apple.com/guide/terminal/open-or-quit-terminal-apd5265185d-f365-44cb-8b09-71a064a42125/mac) or [Linux terminal](https://itsfoss.com/open-terminal-ubuntu/), you'll be able to use BASH commands to directly comunicate with your machine to do basic operations such as creating, deleting or moving files and directories, as well as runs more complicated automatization processes.
# 
# For a deep dive on how to use BASH check-out this [course chapter](link to bash)

# ### Flying blind? Life outside the Graphical user interface: A textbased adventure introducing the terminal, BASH
# 
# 
# **"The Remote Server Adventure"**
# 
# You are a fresh hire in a neurosience lab tasked with locating and copying data stored on a remote server system using the Linux terminal and BASH.
# 
# You've been provided with a set of instructions and a log-in name for the serversystem. But you didn't manage to communicate that you really have no idea what this bash thingy or a Linux is. With the power of the internet and an old notebook, with the name of the previous intern on it, that you've found under the keyboard of your workstation, you set out for your digital adventure!
# 
# **The notebook contains:**
# a scribbled note at the beginnig reading:
# 
# Grey fields such as `this` contain BASH commands, apparentyl you write them into this terminal thing, hit enter and stuff happens?
# 
# 
# Chapter 1: Explore the File System
# 
#     Use the `ls` command to list the contents of the current directory
#     Use the `ls -l` command to list the contents of the directory in long format
#     Use the `pwd` command to display the current working directory
#     Use the `find` command to search for files and directories
#     Use the `cd` command to change the current directory (you'll also have to provide the "path to the directory", which you're trying to reach, e.g. /home/user/Downloads/)
# 
# Chapter 2: Create and Manage Files and Directories
# 
#     Use the `touch` command to create a new file called "index.txt"
#     Use the `mkdir` command to create a new directory called "data"
#     Use the `cp` command to copy "index.txt" to the "data" directory
#     Use the `mv` command to rename "index.txt" to "README.txt"
#     Use the `rm` command to remove "index.txt"
#     Use the `tar`command to create or extract compressed archive files
# 
# 
# Chapter 3: Retrieve System Information
# 
#     Use the `df` command to show the amount of disk space used and available on the file system
#     Use the `top` command to show the system processes and their resource usage
#     Use the `free` command to show the amount of free and used memory in the system
# 
# Chapter 4: Create and Manage User Accounts
# 
#     Use the `adduser` command to create a new user account
#     Use the `usermod` command to modify the user account's properties
#     Use the `passwd` command to change the user's password
#     Use the `deluser` command to delete the user account
# 
# 
# **Your Objective:**
#  
#     Find out where on the file system you are
#     Create a directory to store your copied data in
#     Navigate through the file system
#     Locate the data in question
#     Copy the data into your newly created directory
#     

# #### Excercise:  Start
# 
# Thankfully the IT-person left this "terminal" open. You have just connected to the remote server using ssh <user>@<remote_server_IP>. You see the following line on the screen:
# 
# [username@remote_server ~]$ 
# 
# You decide to try the `pwd` command scribbled in the notebook and hit enter:
# 
# A message appears on the screen reading:
#  
#  /remote/user/
#  
# That's probably the current directory your in, right?
# 
# What's your next objective again? (Use the cells below to fulfill your task 

# In[ ]:





# ### Neuroscience spefiic cloud computing service:
# 
# Brainlife is a platform for sharing, running, and reproducing neuroimaging research. Brainlife provides access to  tools and pipelines created by the neuroimaging comunity to automate certain tasks such as preprocessing MRI data.
# 
# Using Brainlife for MRI preprocessing would look something like this:
# 
#     Create an account on the Brainlife website (brainlife.io)
#     Search for MRI preprocessing tools that meet your specific needs and requirements, e.g. XX
#     Upload your data
#     Launch the selected tool or pipeline by specifying the input data and any necessary parameters
#     Wait for the processing to complete and view the results on the platform
#     Download the processed data for further analysis or visualization
# 
# Note that the specific steps and details of using Brainlife for MRI preprocessing will depend on the tool or pipeline you choose, and it is advisable to consult the documentation and guidelines for that specific tool.

# ### Homework, Excercises and ressources
# 
# ### Additional materials
# 
# 
# ## References
# 
# ## TlDR
