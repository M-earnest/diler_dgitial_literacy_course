#!/usr/bin/env python
# coding: utf-8

# 
# # MNE python for EEG analysis

# **What is <a href="https://mne.tools/stable/index.html" title="https://mne.tools/stable/index.html">MNE</a>?**
# 
# 
# 
# <img style="float:left; margin-right:10px; bottom:0px; width:140px; height:75px; border:none;" src="https://img-9gag-fun.9cache.com/photo/a24XN09_700bwp.webp" alt="Py > Matlab">  
# 
# A way for universities to save a lot cash that would otherwise be invested into matlab licenses, probably,
#  <br>
#  <br>
#  <br>
#  <br>
#  
# But more generally speaking MNE is an open-source Python package for working with MEG, EEG, NIRS, with extensive <a href="https://mne.tools/stable/overview/index.html" title="https://mne.tools/stable/overview/index.html">documentation</a>, <a href="https://mne.tools/stable/auto_tutorials/index.html" title="https://mne.tools/stable/auto_tutorials/index.html">tutorials</a>
#  and <a href="https://mne.tools/stable/python_reference.html" title="https://mne.tools/stable/python_reference.html">API references</a>. The package has an active and engaged developer community on <a href="https://github.com/mne-tools" title="https://github.com/mne-tools">Github</a>, as well as in the <a href="https://mne.discourse.group/" title="https://mne.discourse.group/">discourse forum</a>, where you can turn to with problems, more complex questions or analysis strategies that the provided tutorials may not adress (after reading the  <a href="https://mne.tools/stable/overview/faq.html#faq" title="https://mne.tools/stable/overview/faq.html#faq/">FAQ</a>, of course). If any of the terms used in this tutorial are unclear you can further check out the <a href="https://mne.tools/stable/glossary.html#" title="https://mne.tools/stable/glossary.html#">glossary</a>.
# 
# 
# 
# MNE should provide anything you'd need for for exploring, visualizing, and analyzing neurophysiological data.

# ## Basics
# 
# Let's jump right in. We'll be dealing with EEG data, but the same syntax should apply for most other data supported by MNE. <br>
# We're first gonna import the necessary libraries for this section.

# In[1]:


# import necessary libraries/functions
import os  # for directory navigation
import mne
from mne_bids import BIDSPath, write_raw_bids, print_dir_tree, make_report, read_raw_bids
import pandas as pd  # mostly for saving data as csv 

# this allows for interactive pop-up plots
#%matplotlib qt

# allows for rendering in output-cell
get_ipython().run_line_magic('matplotlib', 'notebook')


# ## Loading data

# 
# MNE uses the <a href="https://mne.tools/stable/generated/mne.io.read_raw.htmls" title="https://mne.tools/stable/generated/mne.io.read_raw.htmls">mne.io.read_raw_*</a>() function to import raw data. The MNE standard is the .fif dataformat, but how data is stored/generated is generally dependent on the software used to record the EEG. MNE provides a list of supported dataformats and the corresponding version of the .read_raw() function you'll to use <a href="https://mne.tools/stable/overview/implementation.html#data-formats">here</a>. For information on how to import data from different recoding systems see <a href="https://mne.tools/stable/auto_tutorials/io/index.htm" title="https://mne.tools/stable/auto_tutorials/io/index.htm">here</a>.
# 
# Head to the <a href="https://mne.tools/stable/auto_tutorials/raw/10_raw_overview.html" title="MNE tutorials: The Raw data structure: continuous data"> MNE tutorials: The Raw data structure: continuous data </a> for some more in-depth explanations from the developers.

# First we'll download the <a href="https://mne.tools/stable/overview/datasets_index.html" title="https://mne.tools/stable/overview/datasets_index.html">sample dataset</a> using the <a href="https://mne.tools/stable/generated/mne.datasets.sample.data_path.html#mne.datasets.sample.data_path" title="https://mne.tools/stable/generated/mne.datasets.sample.data_path.html#mne.datasets.sample.data_path">mne.datasets.sample.data_path()</a> function. .sample.data_path(~/path_to_data) will download the sample data automatically when the data is not found in the provided path. In the downloaded dataset directory we'll next look for the "sample_audvis_raw.fif" file containing the EEG data in the MEG/sample/ path. 
# 
# <br/>

# In[2]:


# load some sample data for convenience, overwrite previous data
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')


# <br/>
# 
# As you might have guessed, to import the "sample_audvis_raw.fif" file we'll use the <a href="https://mne.tools/stable/generated/mne.io.read_raw_fif.html#mne.io.read_raw_fif" title="https://mne.tools/stable/generated/mne.io.read_raw_fif.html#mne.io.read_raw_fif">mne.io.read_raw_fif()</a> function. Setting the preload parameter to "True" will load the raw data into memory, which is needed for data manipulation but may take up large amounts of memory depending on the size of your raw data.

# In[3]:


raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)


# <br/>
# 
# ## Exploring the raw object
# 
# Now we can explore the raw data object. Using the raw.info attribute, we get info on the contained data, such as the number of channels, the sampling frequency with which the data was collected etc., as well as some metadata (such as the measurement date). For an explanation of the listed Attributes see the <a href="https://mne.tools/stable/generated/mne.Info.html#mne.Info" title="https://mne.tools/stable/generated/mne.Info.html#mne.Info">notes in the API reference</a>.
# 
# 

# In[4]:


raw.info


# <br/>
# 
# The **.info.keys()** function displays a dictionary containing the possible parameters, we can pass to the raw.info object.

# In[5]:


raw.info.keys()


# <br/>
# 
# Using these parameters we can access different parts of the .info, such as the events contained in the data

# In[6]:


raw.info['events']


# <br/>
# 
# or a list of the names of all the included channels.

# In[7]:


raw.info['ch_names']


# <br/>
# 
# To visualize the raw data, the <a href="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.plot" title="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.plot">.plot()</a> function can be used to open an interactive plot. See the notes section of <a href="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.plot" title="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.plot">.plot()</a> for more info how to navigate interactive plots created by MNE. (Note: To view interactive output plots run this chapter as a jupyter notebook or via binder)

# In[8]:


# simple plotting function
raw.plot();


# <br/>
# To display the corresponindg positions of the channels contained in the raw object use the <a href="https://mne.tools/stable/generated/mne.viz.plot_sensors.html" title="https://mne.tools/stable/generated/mne.viz.plot_sensors.html">.plot_sensors()</a> function.

# In[9]:


raw.plot_sensors(show_names=True);


# <br/>
# 
# which looks a bit crowded. Consulting the "Good channels" attribute in the .info reveals that we have both MEG and EEG data in our raw object.

# In[10]:


raw.info


# <br/>
# 
# With the ch_type parameter we can select which channels we want to plot. Conventional nomenclature would dictate that channels are named after their respective position on the skull (see <a href="https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)" title="https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)">International 10–20 system (Wikipedia)</a>), unfortunately for the sample data we've only a numbered list of EEG channels. You may further notice that channels markes as "bad" in the .info are displayed in red.

# In[11]:


raw.plot_sensors(show_names=True, ch_type='eeg');


# <br/>
# Using the "kind" parameter we can further display our channel positions in 3d.

# In[12]:


raw.plot_sensors(show_names=True, ch_type='eeg', kind='3d');


# <br/>
# There are multiple ways to access the data contained in the raw object. The tuple syntax shown below for example.

# In[13]:


type(raw[:])


# In[14]:


raw[:]


# 
# <br/>
# Accessing the raw data this way reveals that the data is organized into two numpy arrays, one containing the amplitude in each channel for each sample and the other the corresponding times, resulting in an (n_chans × n_samps) and an (1 × n_samps) matrix respectively.

# raw[:][0] is the (n_chans × n_samps) matrix containing our EEG-amplitudes

# In[15]:


print(type(raw[:][0]))


# In[16]:


raw[:][0]


# In[17]:


print(raw[:][0]) # 1 array containing (n_chans × n_samps)
print(raw[:][0].shape)


# <br/>
# 
# therefore

# In[18]:


raw[:][0].shape[:][0] == raw.info['nchan']


# <br/>
# and Raw[:][1] contains the corresponding time points

# In[19]:


print(raw[:][1])  # 1 array containing times (1 × n_samps)
print(raw[:][1].shape) 


# <br/>
# 
# **Although the above examples may be great for efficient programming pipelines, let's be more specific to see what we are actually dealing with.**
# 
# Let's check out the channels named in the info object again.

# In[20]:


raw.info['ch_names']


# <br/>
# 
# using the ch_names list of our info object, we can access data explicitly by channel name

# In[21]:


print(raw[['EEG 054'], :1000]) # access arrays for channel EEG 054 for the first 1000 samples


# <br/>
# 
# or for multiple channels.

# In[22]:


print(raw[['EEG 054', 'EEG 055'], 1000:2000])


# <br/>
# 
# Referencing specific arrays containing our amplitudes corresponding to the respective channels can again be achieved via tuple indexing

# In[23]:


print(raw[['EEG 054', 'EEG 055'], 1000:2000][0][0]) # access data in channel 'EEG 054' for sample 1000 to 2000


# which is the same as

# In[24]:


print(raw[['EEG 054'], 1000:2000][0])


# In[25]:


# check if the the abpve example truly accesses the same data
print(raw[['EEG 054'], 1000:2000][0] == raw[['EEG 054', 'EEG 055'], 1000:2000][0][0])


# <br/>
# 
# the same applies to our time samples

# In[26]:


# let's take a look at how to access the corresponding time-samples
print('number of time samples')
print(' ')
print(raw[['EEG 054', 'EEG 055'], 1000:2000][1].shape)

print(' ')
print('-------------------------------------------------------')

print('time for sample in array')
print(' ')
print(raw[['EEG 054', 'EEG 055'], 1000:2000][1]) # access corresponding time points


# <br/>
# 
# **We could instead further use the build_in <a href="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.get_data">get_data() </a> function to access data, which returns our n_chan x n_samples matrix and if we set return_times=True the corresponding time-points.**

# In[27]:


data, times = raw.get_data(return_times=True)
print(data.shape)


# In[28]:


data # channels x samples


# In[29]:


times # 1 x samples


# 
# ## Save/export Data
# 
# We can export our data in a number of ways for further analysis, that we'll explore in the following paragraphs.

# ### BIDS data format
# 
# It's advisable to adapt common standards for data organisation, so as to make collaboration, data and code sharing and automated analysis easier. For this we'll use the BIDS data structure, which specifices which file formats to use, file naming conventions and the structure of your data directories.
# 
# More info on the BIDS specification can be found <a href="https://bids-specification.readthedocs.io/en/stable/" title="https://bids-specification.readthedocs.io/en/stable/">here</a>.
# 
# Specific tutorials for working with MNE-BIDS can be found  <a href="https://mne.tools/mne-bids/stable/index.html" title="https://mne.tools/mne-bids/stable/index.html">here</a>, including <a href="https://mne.tools/mne-bids/stable/use.html" title="https://mne.tools/mne-bids/stable/use.html">tutorials</a> that should cover all of the possible use-cases.
# 
# 

# **One of the most important things is to understand the <a href="https://mne.tools/mne-bids/stable/generated/mne_bids.BIDSPath.html" title="https://mne.tools/mne-bids/stable/generated/mne_bids.BIDSPath.html">BIDSPath object</a>**
# 
# 
# The most important components are the root directory and our identifiers, i.e. the subject number, task name, session and run number. So we'll declare those in the next step.

# In[30]:


homedir = os.path.expanduser("~")  # find home diretory
print(homedir)

root = homedir + (os.sep) +  "msc_05_example_bids_dataset"  # string denoting where your directory is supposed to be generated
subject = 'test'  # convention is a 3 numer identifier i.e 001
task = 'audiovisual'
session = '001'
run = '001'


#  Let's set our path

# In[31]:


bids_path = BIDSPath(root=root,  # set BIDSPath
                     subject=subject,
                     task=task,
                     session=session,
                     run=run)


# In[32]:


bids_path


# and use the <a href="https://mne.tools/mne-bids/stable/generated/mne_bids.write_raw_bids.html" title="https://mne.tools/mne-bids/stable/generated/mne_bids.write_raw_bids.html">write_raw_bids()</a> function to save raw data to a BIDS-compliant folder structure.

# In[33]:


write_raw_bids(raw,
               bids_path=bids_path)


# <br/>
# 
# which will result in an error as we've loaded our data into memory. MNE-BIDS <a href="https://mne.tools/mne-bids/stable/generated/mne_bids.write_raw_bids.html" title="https://mne.tools/mne-bids/stable/generated/mne_bids.write_raw_bids.html">write_raw_bids()</a> provides a explicit warning for this case:
# 
# _"BIDS was originally designed for unprocessed or minimally processed data. For this reason, by default, we prevent writing of preloaded data that may have been modified. Only use this option when absolutely necessary: for example, manually converting from file formats not supported by MNE or writing preprocessed derivatives. Be aware that these use cases are not fully supported."_

# So it's advisable to convert your data to the BIDS format beforehand. For this tutorial we'll just import our data again, without loading it into memory.

# In[34]:


raw = mne.io.read_raw_fif(sample_data_raw_file, preload=False)


# In[35]:


write_raw_bids(raw,
               bids_path=bids_path)


# <br/>
# and let's generate an output path from that

# In[36]:


output_path = os.path.join(str(bids_path.root) + (os.sep) + 'sub-test/ses-001/meg')
print(output_path)


# <br/>
# 
# **Now you could simply use the <a href="https://mne.tools/mne-bids/stable/generated/mne_bids.read_raw_bids.html" title="https://mne.tools/mne-bids/stable/generated/mne_bids.read_raw_bids.html">mne_bids.read_raw_bids()</a> function to import data simply by providing the bids_path.**

# In[37]:


raw_bids = read_raw_bids(bids_path=bids_path)


# In[38]:


raw_bids.info


# <br/>
# An important aspect when you're planning on uploading or sharing data with others that we did not have time to adress in this tutorial is annonymization. Consult the <a href="https://mne.tools/mne-bids/stable/auto_examples/anonymize_dataset.html#sphx-glr-auto-examples-anonymize-dataset-py" title="https://mne.tools/mne-bids/stable/auto_examples/anonymize_dataset.html#sphx-glr-auto-examples-anonymize-dataset-py"> MNE-BIDS tutorial: Anonymizing a BIDS dataset </a> for more info.
#     
# More info on the BIDS structure will also be provided in the following sessions/chapters.

# <br/>
# 
# ### Saving Data using raw.save()
# 
# **Now that our directory structure is created we can directly save our raw data using the  <a href="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.save" title="MNE API: mne.raw.save()"> raw.save() </a> function.**
# 
# This would mostly be used after manipulating the data in some way, but we can also select only a subset of channels or time-points. Let's say we're only interested in the contained eeg-channels (+eog-channels for artifact correction later on) for some 30 seconds.

# In[39]:


# have to recur a few folder steps due to raw.save using the absolute path of the current notebook as self-reference
save_path = (('..' + str(os.sep) +
 '..' + str(os.sep) + 
 '..' + str(os.sep) + 
 '..' + str(os.sep) + 
 '..' + str(os.sep) + 
 '..' + str(os.sep) + 
 '..' + str(os.sep) + 
 '..' + str(os.sep) + '..') + output_path)


# save to fif
raw.save(fname=(save_path + str(os.sep) + 'sub-test_ses-001_eeg_data_cropped.fif'),  # the filename
         verbose='INFO',
         picks=['eeg', 'eog', 'stim'],  # the channels to pick (could also be single channels)
         tmin=30,  # first time-point in seconds included 
         tmax=60,  # last time-point included
         overwrite=True)  # checks if data is already present and overwrites if true


# <br/>
# 
# ### Saving data as a csv/tsv using pandas
# 
# We can further export the data to a pandas DataFrame and save the created DataFrame as a tsv file, if you want the data in a more human readable format or want to do your statistics or plotting with another package.
# For more info on how to specify the structure and composition of the DataFrame see <a href="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.to_data_frame" title="MNE API: mne.raw.to_data_frame()"> raw.to_data_frame() </a>

# In[40]:


raw_eeg_df = raw.to_data_frame(picks=['eeg', 'eog'],  # channels to include
                                 index='time',  # set time as index, if not included time will be a separte column
                                 start=30,  # first sample to include
                                 stop=60,  # first sample to include
                                 long_format=False)  # specify format


# In[41]:


raw_eeg_df


# <br/>
# For some analysis is preferable to have data organised in long format, i.e. so that everey sample for every channel is one row. This can be achieved by setting the long_format parameter to "True".

# In[42]:


raw_eeg_df = raw.to_data_frame(picks=['eeg', 'eog', 'stim'],  # channels to include
                                 index='time',  # set time as index, if not included time will be a separte column
                                 start=30,  # first sample to include
                                 stop=60,  # first sample to include
                                 long_format=True)  # specify format



# In[43]:


raw_eeg_df


# <br/>
# 
# **To save our DataFrame, we can now simply use the pands fucntion <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html" title="pandas.DataFrame.to_csv()"> pd.DataFrame.to_csv() </a>**

# In[44]:


fname = output_path + str(os.sep) + 'sub-test_ses-001_eeg_data_cropped.tsv'
raw_eeg_df.to_csv(fname, sep='\t')


# ## Further reading
# 
# For some more in-depth explanations on how to work with raw data, such as how to drop or rename channels, see <a href="https://mne.tools/stable/auto_tutorials/raw/10_raw_overview.html" title="MNE tutorials: The Raw data structure: continuous data"> MNE tutorials: The Raw data structure: continuous data </a>.
# 
# For an overview of the general workflow using MNE, see the <a href="https://mne.tools/stable/overview/cookbook.html" title="MNE cookbook">MNE cookbook</a>.
