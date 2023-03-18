#!/usr/bin/env python
# coding: utf-8

# # Data-mangament
# 
# ## General introduction
# The topic of data managment can seem quite trivial for beginners, but starting out with best practices will make your future significantly easier and help ensure your data is well-organized and secure.

# ### Where are we now?
# 
# Before we start a lesson it is usually helpfulto reflect on what we already know about a certain topic or e.g. what a lesson may possibly try to teach us.
# 
# So please take a few minutes to reflect on the concept of digital literacy with the following questions.
# 
# ----- ToDO: rephrase/expand
# 
# **1.1 What is your understanding of data managment?**
# 
# **1.2 What could you possibly be learn?**
# 
# **1.3 How do you usually store/manage data?**
# 
# 
# 
# _Note: Feel free to do this in your head or in a separate document. Remember, to interactively engage with the material either open it in MyBinder (the small rocket button at the top of the website) or [download the course material](link to course zip), go through the [setup process](link to setup) and open this file (i.e digital_literacy.ipynb in the introduction folder) using Jupyter Notebooks or VScode._

# ## Goals
# 
# ----- ToDO: adapt & expand
# 
# Specific goals of this first session will be to, e.g.
# 
# * general understanding
# * get familiar with process
# * provide a checklist to follow
# * understanding on why project design is an essential step

# ## Roadmap
# 
# 
# - **Goals**
# - Data managment
#     - 1. Data managment plan
#     - 2. Setup local folder structure (BIDS)
#     - 3. Research Checklist
# - Data storage
#     1. Open Brain Consent Form/GDPR & you
#     2. licensing
#     3. Open data
#     4. Connecting to an online repository
#     5. Backup procedures
# 
# 

# # Data Managment
# 
# ## 1. Data managment plan
# 
# 
# An initial step when starting any research project should be to setup a data managment plan.
# 
# This helps us to flesh out, describe and doucment what data exactly we want to collect, what we'll be doing with it, where and how it's stored and eventually shared.
# 
# 
# ---- ToDO: write intro/description &  what a data managment plan consists of
# 
# 
# - consider how the data will be used, archived and shared
# 
# 
# #### Motivation?
# 
# For the public good
# - if widely adopted makes it inherently more easier to reproduce code and anylsis pipelines build by others, therefore lowering scientific waste and improving efficiency
# 
# For yourself
# 
# - You are likely the future user of the data and data analysis pipelines you’ve developed, so keeping your file structure standarized removes the need to remember where you've stored specific pieces of data etc.
# - Enables and simplifies collaboration; - allows reader/collarorators to gain a quick understanding of what data you'll be collecting, where this data can be found and what exactly you're planing on doing with it
# - Reviewers and funding agencies like to see clear, reproducible results
# - Open-science based funding opportunities and awards available (for instance: OHBM Replication Award, Mozilla Open Science Fellowship, Google Summer of Code, and so on.)
# 
# ### FAIR principles
# 
# ----- ToDO: expand & explain motivation, add example
# 
# 
# **FAIR principles**: The FAIR principles stand for Findable, Accessible, Interoperable, and Reusable. These principles aim to make research data more accessible and reusable by promoting the use of standardized metadata, persistent identifiers, and open data formats. Allowing research not only to be shared, but also to be actually found.
# 
# - https://the-turing-way.netlify.app/reproducible-research/rdm/rdm-fair.html
# 
# 
# ### What to consider in your data management plan
# 
# ----- ToDO: expand
# 
# 
# - e.g. Turing way (https://the-turing-way.netlify.app/reproducible-research/rdm.html#rr-rdm)
#  
# 1. Roles and Responsibilities
# 2. Type and size of data collected and documentation/metadata generated
# 3. Type of data storage used and back up procedures that are in place
# 4. Preservation of the research outputs after the project
# 5. Reuse of your research outputs by others
# 6. Costs
# 
# 
# 
# 
# ### DMP tools
# 
# ----- ToDO: expand & explanation of what dmp tools do
# https://argos.openaire.eu/home
# 
# 
# 
# 

# 
# ##  2. Setup local folder structure 
# 
# 
# It is recommended to adopt a standarized approach to structuring your data, as this not only helps you stay consistent, but also allows you and possible collaborators to easily identify where specific data is located.
# 
# 
# ### general file naming conventions
# 
# To make sure that it is easily understood what a file contains and to make files easier for computers to process, you should follow certain naming conventions:
# 
# 
#     - be consistent
#     - use the date in the format YYYYMMDD
#     - use underscores `(_)` instead of spaces or
#     - use camelCase (capitalized first letter of each word in a phrase) instead of spaces
#     - avoid spaces, special characters `(+-"'|?!~@*%{[<>)`,  punctuation `(.,;:)`, slashes and backslashes `(/\)`
#     - avoid "version" names, e.g. v1, vers1, final, final_really etc. (instead use a version control system like github)
# 
# - [MIT cheatsheet for file naming conventions](https://www.dropbox.com/s/ttv3boomxlfgiz5/Handout_fileNaming.pdf?dl=0)
# 
# 
# ### Documentation & Meta-data
# 
# --- ToDO:
#     - description/intro
#     - why it matters
#     - digital tools
#     - what to inculde
#     - document names (REAMDE etc.)
#         Documenting your naming and organization schema, sample
# 
# 
# 
# ### Establish a folder hierarchy
# 
# Before you begin working on your project you should already start setting up the local folder structure on your system. This helps you keep organized and saves you a lot of work in the long run. 
# 
# Your folder hierarchy of course depends on your projects specific need (e.g. folders for data, documents, images etc.) and should be as clear and consistent as possible. The easiest way to achieve this is to copy and adapt an already existing folder hierarchy template for research projects.
#     
#     
# One for example (including a template) is the [Transparent project management template for the OSF plattform](https://osf.io/4sdn3/) by [C.H.J. Hartgerink](https://osf.io/5fukm/)
# 
#    
# The contained folder structure would then look like this:
# 
# ```
# project_name/
#     └── archive
#     │   └── 
#         
#     └── analyses
#     │   └── 
#     │   
#     └── bibliography
#     │   └── 
#     │   
#     └── data
#     │   └── 
#     │   
#     └── figure
#     │   └── 
#     │   
#     └── functions
#     │   └── 
#     │   
#     └── materials
#     │   └── 
#     │   
#     └── preregister
#     │   └── 
#     │
#     └── submission
#     │   └── 
#     │   
#     └── supplement
#         └── 
# ```   
# 
# 
# 
# Another example would be the ["resarch project structure"](http://nikola.me/folder_structure.html) by [Nikola Vukovic](http://nikola.me/#home)
# 
# Where the folder hierarchy would look like this:
#        
#        
# </br>
# 
# 
# ```
# project_name/
#     └── projectManagment/
#     │   ├── proposals/
#     │   │        └── 
#     │   ├── finance/
#     │   │       └── 
#     │   └── reports/
#     │           └── 
#     │   
#     └── EthicsGovernance
#     │   ├── ethicsApproval/
#     │   │       └── 
#     │   └── consentForms/
#     │           └── 
#     │   
#     └── ExperimentOne/
#     │   ├── inputs/
#     │   │       └── 
#     │   ├── data/
#     │   │       └── 
#     │   ├── analysis/
#     │   │       └── 
#     │   └── outputs/
#     │           └── 
#     │   
#     └── Dissemination/
#         ├── presentations/
#         │       └── 
#         ├── publications/
#         │       └── 
#         └── publicity/
#                 └── 
# ```   
# 
# 
# 
# </br>
# 
# 
# </br>
# 
# 
# 
# ### Incorporating experimental data/BIDS standard
# 
# Now both of these examples provide an "experiment folder", but tend to utilize/establish their own standards. 
# 
# But we aim to make our folder structure as easily understandable, interoperable (e.g. between systems and programms) and reproducible, therefore it is best to adapt our "experiment folder" to industry standards.
# 
# For most experimental data the most promising approach to this is [BIDS](https://bids.neuroimaging.io/) (Brain Imaging Data Structure). Originally conceptualizes as a standarized format for the organization and description of fMRI data, the format can be extended to encompass other kinds of neuroimaging and behavioral data. Using the BIDS standard makes the integration of your data into most neuroscience analysis pipelines 
# 
# 
# - bids.jpg
# 
# 
# ### Bids quick introduction
# 
# ---- ToDo: describe and explain names and what files contain (e.g. metadata, tsvs, jsons)
# 
# - in theory
# 
# ```
# project/
#     ├── derivatives/
#     ├── code/
#     └── subject/
#         └── session/
#             └── datatype
# ```     
#         
# - in practice   
#         
# ```
# project_01
# ├── dataset_description.json
# ├── participants.tsv
# ├── derivatives
# ├──code
# ├── sub-01
# │   ├── anat
# │   │   ├── sub-01_inplaneT2.nii.gz
# │   │   └── sub-01_T1w.nii.gz
# │   └── func
# │       ├── sub-01_task-balloonanalogrisktask_run-01_bold.nii.gz
# │       ├── sub-01_task-balloonanalogrisktask_run-01_events.tsv
# │       ├── sub-01_task-balloonanalogrisktask_run-02_bold.nii.gz
# │       ├── sub-01_task-balloonanalogrisktask_run-02_events.tsv
# │       ├── sub-01_task-balloonanalogrisktask_run-03_bold.nii.gz
# │       └── sub-01_task-balloonanalogrisktask_run-03_events.tsv
# ├── sub-02
# │   ├── anat
# │   │   ├── sub-02_inplaneT2.nii.gz
# │   │   └── sub-02_T1w.nii.gz
# │   └── func
# │       ├── sub-02_task-balloonanalogrisktask_run-01_bold.nii.gz
# │       ├── sub-02_task-balloonanalogrisktask_run-01_events.tsv
# │       ├── sub-02_task-balloonanalogrisktask_run-02_bold.nii.gz
# │       ├── sub-02_task-balloonanalogrisktask_run-02_events.tsv
# │       ├── sub-02_task-balloonanalogrisktask_run-03_bold.nii.gz
# │       └── sub-02_task-balloonanalogrisktask_run-03_events.tsv
# ...
# ...
# └── task-balloonanalogrisktask_bold.json
# ```
# 
# To learn more checkout the following ressources:
# 
# https://bids-standard.github.io/bids-starter-kit/folders_and_files/folders.html
# 
# 
# 
# 
# 
# ### Finished folder structure
# 
# As long as you adapt you "experiment" folder to the bids standarad the rest is up to your preference. Therfore your finished folder structure could look something like this:
# 
# ```
# project_name/
#     └── projectManagment/
#     │   ├── proposals/
#     │   │        └── 
#     │   ├── finance/
#     │   │       └── 
#     │   └── reports/
#     │           └── 
#     │   
#     └── EthicsGovernance
#     │   ├── ethicsApproval/
#     │   │       └── 
#     │   └── consentForms/
#     │           └── 
#     │   
#     ├── project_bids/
#     │   ├── derivatives/
#     │   ├── code/
#     │   └── subject/
#     │       └── session/
#     │           └── datatype
#     │   
#     └── Dissemination/
#         ├── presentations/
#         │       └── 
#         ├── publications/
#         │       └── 
#         └── publicity/
#                 └── 
# ```   

# 
# ## 3. Research Checklist:
# 
# ----- ToDO: expand and explain motivation
# 
# - https://the-turing-way.netlify.app/reproducible-research/rdm/rdm-checklist.html

# ## Data storage:
# 
# Now while having a standarized data format on our system, let's turn towards the topic of data storage. While for small projects or purely behavioral research your local file system may seem initially sufficient, for larger datasets and file sizes, such as in neuroimaging, you'll quickly run out of storage. The same applies when you collect a larger number of smaller projects throughtout your career, of course. Further, do we really want to store data simply on some laptop until it is forgotten/deleted and in general inaccessible to others who might make use of it?
# 
# (Un/)Fortunately sharing data falls under the jurisdiction of local laws, e.g. the General Data Protection Regulation (GDPR) in germany. It is therfore essential to make sure that where and how you store or share the data you will be collecting or have already collected is in compliance with the law.
# 
# 
# ### 1. Open Brain Consent Form/GDPR & you
# 
# Research especially in pschology or neuroscience is often dependent on the collection of new data from human subjects. Given the complexities of the (german) General Data Protection Regulation [(GDPR)](https://gdpr-info.eu/) and to avoid tassles with the local ethics comittees the "Open Brain Consent Form" was created to be ethically and legally waterproof, while allowing researchers to openly share their collected datasets. While the document was developed for neuroimaging studies it can be adapted to most forms of data with little effort.
# 
# 
# 
# **As we are comitted to research transaprency and open sciene we advise you to make use of this document for your own research.**
# 
# [Make open data sharing a no-brainer for ethics committees](https://open-brain-consent.readthedocs.io/en/stable/index.html).
# 
# **GDPR conform:**
# 
# [German version](https://open-brain-consent.readthedocs.io/en/stable/gdpr/i18n/ultimate_gdpr.de.html)
# 
# [English version (GDPR)](https://open-brain-consent.readthedocs.io/en/stable/gdpr/ultimate_gdpr.html#english)
# 
# **Outside of germany:**
# 
# [Ultimate brain consent form](https://open-brain-consent.readthedocs.io/en/stable/ultimate.html)
# 
# 
# **When using this template please acknowledge the creators accordingly:**
# 
# Open Brain Consent working group (2021). The Open Brain Consent: Informing research participants and obtaining consent to share brain imaging data. Human Brain Mapping, 1-7 https://doi.org/10.1002/hbm.25351.
# 
# 
# ### 2. licensing
# 
# 
# https://the-turing-way.netlify.app/reproducible-research/licensing/licensing-data.html
# 
# 
# 
# 
# ### 3. Open data
# 
# 
# - general intorudction + reminder open science/access
# - incorporate links
# 
# 
# 
# Nowadays there are a lot of cloud storage options, which additionally allow researchers to share their data given certain rules and restrictions. The following is an incomplete, but somewhat exhaustive list of the possible storage options you face.
# 
# 
# **Open Science Repositories**: Repositories such as OSF, OpenNeuro, the OpenfMRI database, and the IEEE DataPort provide open access to MRI and EEG datasets, as well as other neuroimaging data.
# 
# **U.S. National Institutes of Health**: The National Institutes of Health (NIH) of the U.S. provides open access to many MRI and EEG datasets through their National Library of Medicine's National Institutes of Health Data Sharing Repository.
# 
# **Research Data Repositories**: Zenodo, Figshare, and other research data repositories allow scientists to store, share, and publish their data in an open and transparent manner. These repositories are often committed to open access principles and provide a centralized location for data and metadata, as well as version control and preservation features.
# 
# **Research Collaborations**: Collaborative projects, such as the Human Connectome Project, the International Neuroimaging Data-Sharing Initiative (INDI), and the ABIDE (Autism Brain Imaging Data Exchange) project, provide open access to large datasets collected from multiple sites.
# 
# **Domain-Specific Repositories**: There are also domain-specific repositories for various scientific fields, such as the NCBI Sequence Read Archive for genomics data, the European Geosciences Union Data Centre for Earth and Environmental Science data, and the International Astronomical Union's SIMBAD database for astronomical data. These repositories often have specific requirements for data deposition and sharing, but provide a dedicated space for researchers to share their data in an open and transparent manner.
# 
# 
# 
# ### 4. Connecting to an online repository
# ----- ToDO: quick tutorial
# 
# - OSF
# - Github
# - link back to in-depth tutorials in collaboration & communication
# 
# 
# ### 5. Backup procedures
# 
# ----- ToDO: expand
# 
#         - description
#         - why 
#         - how
#             - cloud storage
#             - server system
#             - harddrives
#         - frequency

# ### Homework, Excercises and ressources
# 

# ### Additional materials
# 
# 
# ## References
# 
# 

# ### Acknowledgments:
# 
# - Turing way
# - BIDS community
# 
#     
# 

# ### Where are we now?
# 
# ---- adapt and expand ------
# 
# Please take a moment to reflect on if and how your understanding of the concept of digital literacy has possibly changed following this lecture.
# 
# **2.1 How well informed we're you with regard to the current topic? (i.e.,
# did you know what was expecting you?)**
# 
# **2.2) What was the most surprising part of the training/lecture for you? What information in particular stood out to you?**
# 

# ## TLDR
