#!/usr/bin/env python
# coding: utf-8

# # EEG preprocessing with MNE

# <br/>
# 
# Preprocessing describes the process of preparing neurophysiological data, such as the EEG data we'll be dealing with in this chapter, for further analysis. This mainly involves removing artifacts (interference or noise due to physiological or environmental) from the recorded data, cutting out breaks in experiments, removing faulty (bad) channels and dividing the continous data into chunks centered around certain events (e.g. the presentation of a stimulus, participant's reactions etc.). For the analysis of EEG data a few extra-steps are necessary to produce a dataset that can provide precise information about changes in neural activity or localization of involved areas, such as setting up a reference channel and applying a montage.
# 
# **For examples of possible preprocessing steps see:**
# 
# <a href="https://www.frontiersin.org/articles/10.3389/fninf.2015.00016/full#h3">The PREP pipeline: standardized preprocessing for large-scale EEG analysis </a>
# 
# 
# <a href="https://www.youtube.com/watch?v=JMB9nZNGVyk">Youtube: Overview of possible preprocessing steps (Mike Cohen)
#  </a>
#  
# - and corresponding a whole <a href="https://mikexcohen.com/lectures.html">set of lectures for EEG data analysis (Mike Cohen)
#  </a>
#  
# 
# **There is also an automated MNE pipeline that is continously improved**
# 
# <a href="https://mne.tools/mne-bids-pipeline/getting_started/basic_usage.html">
# MNE-BIDS-Pipeline  </a>
# 
# 

# <br/>
# 
# ## Setup
# 
# As in the previous chapter we'll first import our necessary libraries and the <a href="https://mne.tools/stable/overview/datasets_index.html" title="https://mne.tools/stable/overview/datasets_index.html">sample dataset</a> provided by MNE. 

# In[1]:


# import necessary libraries/functions
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

# this allows for interactive pop-up plots
#%matplotlib qt

# allows for rendering in output-cell
get_ipython().run_line_magic('matplotlib', 'inline')

# import mne library
import mne
from mne_bids import BIDSPath, write_raw_bids, print_dir_tree, make_report, read_raw_bids


# In[2]:


# load some sample data, overwrite previous data
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)


# In[3]:


raw.info  # check the info object


# <br/>
# 
# ## 1.Referencing

# As the EEG is a measurement of voltage fluctuations (differences in electric potential), we will have to naturally compare the signal measured at a specific electrode to that measured at another location. 

# Therefore we'll first define a reference channel, i.e. an electrode with which the signal of electrodes of interest can be compared. This is effecitively done by substracting the signal of the reference from our measurement electrodes. When recording EEG a specific electrode or a set of electrodes should be employed that are placed in such a way that they collect ideally all the "noise" that is also influencing our measurement electrodes at the scalp, but as little as possible of the actual neural activity of interest. Common choices for the placement of reference electrodes are the earlobes or the mastoids (the bone ridges behind the ears), although sometimes central electrodes such as Cz or FCz are chosen. If we'd be interested in the signal measuered at these sites, we would have to re-reference our data.
# 
# There is a few ways to go about referencing, chek-out the <a href="https://mne.tools/stable/auto_tutorials/preprocessing/55_setting_eeg_reference.html" title="https://mne.tools/stable/auto_tutorials/preprocessing/55_setting_eeg_reference.html">MNE tutorial: Setting the EEG reference</a> for more info.

# First we'll exclude the meg channels from our data, as we won't be needing those for our EEG introduction. We'll use the <a href="https://mne.tools/stable/generated/mne.pick_types.html" title="https://mne.tools/stable/generated/mne.pick_types.html">pick_types()</a> fucntion, specifying which channels to keep and which to exclude by type.

# In[4]:


raw = raw.pick_types(meg=False, eeg=True, eog=True, stim=True)


# <br/>
# 
# We can use use the <a href="https://mne.tools/stable/generated/mne.viz.plot_sensors.html" title="https://mne.tools/stable/generated/mne.viz.plot_sensors.html">.plot_sensors()</a> function to see how electrodes where placed on the scalp.

# In[5]:


raw.plot_sensors(show_names=True, kind='3d', ch_type='eeg');


# In[6]:


raw.plot(n_channels=61);


# <br/>
# 
# Unfortunately the sample data wasn't collected using one of the <a href="https://www.fieldtriptoolbox.org/getting_started/1020/" title="https://www.fieldtriptoolbox.org/getting_started/1020/">standard layouts</a> we usually see, so it is unclear if any mastoid electrodes we're placed. Corresponding most closely to the usual positions would be the electroed "EEG 025" and "EEG 024", therefore we'll see how these would function as a reference channel using the <a href="https://mne.tools/stable/generated/mne.set_bipolar_reference.html" title="https://mne.tools/stable/generated/mne.set_bipolar_reference.html">mne.set_bipolar_reference()</a> function.
# 

# In[7]:


raw_bi_ref = mne.set_bipolar_reference(raw, 
                                       anode=['EEG 025'],
                                       cathode=['EEG 024'])


# <br/>
# 
# and comparing the referenced data to our raw data, we'll see that the reference channel EEG 025-EEG 024 was added in 

# In[8]:


raw_bi_ref.plot(n_channels=61);


# <br/>
# 
# With the <a href="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.set_eeg_reference" title="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.set_eeg_reference">.set_eeg_reference()</a>  function we can also apply a virtual reference channel by averaging over all electrodes by setting the parameter to ref_channels='average'. This will overwrite our original raw data, therefore we'll make a <a href="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.copy" title="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.copy">copy()</a> of our raw object before assigning the data to a new variable.

# In[9]:


# use the average of all channels as reference
raw_avg_ref = raw.copy().set_eeg_reference(ref_channels='average')
raw_avg_ref.plot();


# <br/>
# 
# We'll be using the averaged referenced data for the next preprocessing steps, so we can get rid of the bipolar reference.

# In[10]:


del raw_bi_ref  # delete the bipolar referenced data as we won't be using it


# <br/>
# 
# ## 2. Artifact detection

# **The 3 main sources of artifacts are environmental (due to power lines or electroncis), instrumentation (faulty electrodes), and biological (eye-movement, muscular activity).**
# 
# For an extensive list see the <a href="https://mne.tools/stable/auto_tutorials/preprocessing/10_preprocessing_overview.html#sphx-glr-auto-tutorials-preprocessing-10-preprocessing-overview-py" title="https://mne.tools/stable/auto_tutorials/preprocessing/10_preprocessing_overview.html#sphx-glr-auto-tutorials-preprocessing-10-preprocessing-overview-py">MNE tutorial on artifact detection</a> 
# 
# The easiest way to spot these artifacts is to simply plot the raw data if you know what you're looking for.
# For example, artifacts due to eye movement will appear as a sudden step in amplitude that persist for a while in the raw data. While artifacts due to blinking will appear as a sudden spike in amplitude that is strongest in frontal electrodes and subsides with distance to the eyes. Artifacts due to muscle activity appear as sudden bursts of high frequency activity with high amplitude in the raw data.
# 
# Let's see what we can find

# In[11]:


raw_avg_ref.plot();


# <br/>
# 
# Another possibility is to plot the power spectral density (PSD, i.e. the power as µV²/Hz in dB in the frequency domain) of of our signal for each channel using the <a href="https://mne.tools/stable/generated/mne.viz.plot_raw_psd.html" title="MNE API: raw.plot_psd()"> raw.plot_psd() </a> function.
# 
# The clear spikes in power in the 60 Hz, 120 Hz etc. frequeny range are indicative of environmental interference caused by AC power lines.

# In[12]:


raw_avg_ref.plot_psd(spatial_colors=True, picks=['eeg']);


# <br/>
# 
# To identify artifacts automatically, we can use specific MNE functions.
# 
# E.g. for finding eye-movemnt artifacts due to blinks MNE provides <a href="https://mne.tools/stable/generated/mne.preprocessing.create_eog_epochs.html" title="MNE API: create_eog_epochs"> create_eog_epochs() </a> and <a href="https://mne.tools/stable/generated/mne.preprocessing.find_eog_events.html" title="MNE API: find_eog_events"> find_eog_events() </a>, which we'll use to illustrate the mean wave-form of blinks in our data:

# In[13]:


eog_epochs = mne.preprocessing.create_eog_epochs(raw_avg_ref)  # create eog epochs
eog_epochs.average().plot_joint();   #  plot eog activity in source space and topography


# <br/>
# 
# **Bad channels are most easliy identified by  visual inspection:**
# Does the signal captured by an electrode look significantly different in relation to surrounding channels? E.g. a completly flat line, high variability (i.e. appear "noisy") or slow amplitude "drifts" over prolonged timespans. Calling the .plot() function allows us to mark channels that appear problematic by simply clicking on them, which will add them to a list object containing info on all channels marked as bad.

# In[14]:


raw_avg_ref.plot();


# <br/>
# 
# For a more automatic approach we can identify bad channels if the contained signal exceeds a certain deviation threshold by calculating the median absolute deviation of the signal in each channel compared to the others (check <a href="https://cloudxlab.com/assessment/displayslide/6286/robust-z-score-method" title="https://cloudxlab.com/assessment/displayslide/6286/robust-z-score-method">robust-z-score-method</a> for more info).

# In[15]:


channel = raw_avg_ref.copy().pick_types(eeg=True).info['ch_names']  # get list of eeg channel names
data_ = raw_avg_ref.copy().pick_types(eeg=True).get_data() * 1e6  # * 1e6 to transform to microvolt

# calculate median absolute deviation for each channel
mad_scores =[scipy.stats.median_abs_deviation(data_[i, :], scale=1) for i in range(data_.shape[0])]

# compute z-scores for each channel
z_scores = 0.6745 * (mad_scores - np.nanmedian(mad_scores)) / scipy.stats.median_abs_deviation(mad_scores, 
                                                                                                scale=1)
# 1/1.4826=0.6745 (scale factor for MAD for non-normal distribution) 

print(z_scores)
# get channel containing outliers 
bad_dev = [channel[i] for i in np.where(np.abs(z_scores) >= 3.0)[0]]
print(bad_dev)


# <br/>
# 
# ## 3. Artifact removal/repair
# 
# Next we'll try to eliminate and or repair the previously identified artifacts with using different methods (frequency-filtering, rejecting bad channels and Independent component analysis). It is important to note that not all artifacts are problematic or should necessarily lead to complete data rejection, so we'll try to retain as much signal as possible by removing the contaminated signal component and estimating the true signal we'd record if the specific artifacts where not present.
# 
# For a primer on how to decide which artifacts should be removed/not removed see <a href="https://www.youtube.com/watch?v=VDqwfP0mlfU" title="https://www.youtube.com/watch?v=VDqwfP0mlfU">Youtube: Signal artifacts (not) to worry about</a>.

# <br/>
# 
# ### 3.1 Filtering

# 
# Line noise and slow drifts are most easily dealt with by **applying a filter to our data that artifically removes any activity above (highpass) or below (lowpass) a certain frequency**. As the range of frequency that is actually relevant for our data or plausibly produced by neural activity is rather limited, we can for example get rid of the above identified line noise artifacts quite easily by setting a cut-off frequency (lowpass-filter) for our data somewhere below 60Hz.

# There is extensive literature on and a plethora off methods to consider when it comes to eeg-filters, but for this tutorial we'll be using the mne standard parameters. For more info on eeg-filters see the <a href="https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html" title="MNE: background on filtering">MNE: background on filtering</a> or for a visual approach <a href="https://www.youtube.com/watch?v=2tmilbi4L0o" title="Youtube: FIR Filter Design using the Window Method">Youtube: FIR Filter Design using the Window Method</a>.
# 
# 
# **We'll be setting a highpass filter at 0.1hz (to get rid of slow drifts in electrode conductance over time) and a low-pass filter of 40hz (to get rid of electrical line noise).**
# 

# In[16]:


raw_filtered = raw_avg_ref.copy().filter(l_freq=.1, h_freq=40.,  # lfreq = lower cut-off frequency; hfreq =upper cut-off frequency
                 picks=['eeg', 'eog'],  # which channel to filter by type
                 filter_length='auto',
                 l_trans_bandwidth='auto',
                 h_trans_bandwidth='auto',
                 method='fir',  # finite response filter; MNE default
                 phase='zero',
                 fir_window='hamming',  # i.e. our "smoothing function" (MNE-default)
                 fir_design='firwin',
                 n_jobs=1)


# <br/>
# 
# Comparing the filtered data in the time domain shows that we got rid of a lot of the "noisiness" present in the individual channels, as well as attentuated some of the high-frequency muscle artifacts. Ocular artifacts are still clearly present.

# In[17]:


raw.plot();


# In[18]:


raw_filtered.plot();


# <br/>
# 
# Comparing the power spectral density plots of our filtered and unfiltered data revals that frequencies above the specifed cut-off are basically eliminated from our data. The power-spectral density plot further illustrates that there is a certain drop-off range called the **transition band**, which in this case is about 10Hz after the specified cut-off frequency pf 40Hz

# In[19]:


# unfiltered power spectral density (PSD)
raw_avg_ref.plot_psd(spatial_colors=True);


# In[20]:


# filtered power spectral density (PSD)
raw_filtered.plot_psd(spatial_colors=True);


# <br/>
# 
# ### 3.2 Interpolating bad channels
# 
# Let's first take a look at the previoulsly marked bad channels for our unfiltered data again.

# In[21]:


bad_dev


# <br/>
# 
# Next we calculate the median absolute deviations again, but this time for the filtered data.

# In[22]:


channel = raw_filtered.copy().pick_types(eeg=True).info['ch_names']  # get list of eeg channel names
data_ = raw_filtered.copy().pick_types(eeg=True).get_data() * 1e6  # * 1e6 to transform to microvolt

# calculate median absolute deviation for each channel
mad_scores =[scipy.stats.median_abs_deviation(data_[i, :], scale=1) for i in range(data_.shape[0])]

# compute z-scores for each channel
z_scores = 0.6745 * (mad_scores - np.nanmedian(mad_scores)) / scipy.stats.median_abs_deviation(mad_scores, 
                                                                                                scale=1)
# 1/1.4826=0.6745 (scale factor for MAD for non-normal distribution) 

print(z_scores)
# get channel containing outliers 
bad_dev = [channel[i] for i in np.where(np.abs(z_scores) >= 3.0)[0]]
print(bad_dev)


# <br/>
# Now we see that only 2 out of 5 channels are considered to be outliers. So it appears that our filter got rid of some of the problematic noise in our channels already. We'll check if there is already info about bad channels in the metadata of our raw object, that the experimenters may have already provide. Information on which channels are bad is stored in the .info object. 

# In[23]:


raw_filtered.info['bads']


# <br/>
# 
# Adding our outlier channels to the list of "bad" channesl basically means that we declare that these channels should not be considered for further analysis. 
# 
# As .info['bads'] is a regular python list we can edit it in a few simple ways, e.g. let's say electrode EEG 007 would look wildly out of place compared to the signal in the surrounding electrodes. Now e.g. we can add entries using .append() to our list

# In[24]:


raw_filtered.info['bads'].append('EEG 007')               # add bad channel
raw_filtered.info['bads']


# <br/>
# remove entries via .pop

# In[25]:


raw_filtered.info['bads'].pop(-1)  # remove last entry
raw_filtered.info['bads']


# <br/>
# and add multiple values to our list of bads or even completly replace our list of bad channels.

# In[26]:


old_bads = raw_filtered.info['bads'] # create variable containing original bad channels
new_bads = ['EEG 008', 'EEG 009']  # create list of channels to add

# overwrite the whole list
raw_filtered.info['bads'] = new_bads
print(raw_filtered.info['bads'])

# or combine old an new list of bad channels
raw_filtered.info['bads'] = old_bads + new_bads
print(raw_filtered.info['bads'])


# <br/>
# Let's empty our list of bad channels for now

# In[27]:


raw_filtered.info['bads'] = [] # set empty list
print(raw_filtered.info['bads'])


# and add the outlier channels we've identified above

# In[28]:


raw_filtered.info['bads'] = bad_dev 
print(raw_filtered.info['bads'])


# <br/>
# taking a look at the "Bad channels" row of the .info reveals that the marked channels are now part of the raw objects metadata.

# In[29]:


raw_filtered.info


# <br/>
# 
# ### Interpolation

# 
# Now that we know which channels are problematic, we can **estimate their actual signal given the signal of the surrounding electrodes** via the <a href="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.interpolate_bads" title="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.interpolate_bads">.interpolate_bads()</a> fucntion, as with the relatively poor spatial resolution it is highly unklikey that one electrode would have picked up a signal that is not also to a to lesser degree present in the surrounding electrodes. This concept is illustrated well in the visualizatio of the topopgraphic spread of evoked potentials <a href="https://mne.tools/stable/auto_examples/visualization/topo_compare_conditions.html" title="https://mne.tools/stable/auto_examples/visualization/topo_compare_conditions.html">here</a>. See the <a href="https://mne.tools/stable/auto_examples/preprocessing/interpolate_bad_channels.html#footcite-perrinetal1989" title="MNE tutorial: interpolate bad channels"> MNE tutorial: interpolate bad channels </a> for more on the implementation and the therein cited papers for the technical information.

# When to interpolate or drop bad channels should be decided on an individual basis. In general, **if the signal picked up is consistently bad/noisy and might influence the overall quality of the data the channel can be interpolated**. If there are only single "ariftact events in" the channel it's best to keep the channel and later on exclude the epoch containing the artifact. For automatization purposes you'll of course have to make an a priori decison on what levels of noise e.g. you're willing to accept.

# In[30]:


print(raw_filtered.info['bads'])


# <br/>
# 
# For ilustrative purposes we'll interpolate the channels we've identified as outliners above and compare the data before and after interpolation.

# In[31]:


interp_ = raw_filtered.copy().interpolate_bads(reset_bads=False)  # copy() as this would modify the raw object in place
interp_.plot();


# In[32]:


raw_filtered.plot();


# <br/>
# We see a reduction in the amplitude of the high frequency components in the channels "EEG 004" and "EEG 007", as well as an general "reshaping" of the data as a composition of activity in the surrounding channels. A prominent example would be the change in wave-form for the first blink-artifact in channel "EEG 004".

# Let's apply this to our data and reset our bad channels, so that they'll not be excluded from further analysis anymore.

# In[33]:


raw_filtered.interpolate_bads(reset_bads=True)


# In[34]:


del interp_ # delete to free up space


# <br/>
# 
# ### 3.3 Independent component analysis (ICA)

# 
# **One of the most common and effetive tools for artifact detection and removal is the Independent component analysis, which we'll use to exclude our occular artifacts from the data in the next step**
# 
# In brief:
# ICA is a technique for signal processing that separates a signal into a specified number of linearly mixed sources. For the preprocessing of EEG data this is  used to find statistically independet sources of signal variability present in our n_channel dimensional data. EEG artifacts are usually strong sources of variability as they show higher amplitude than would be expected for sources of neural activity and generally appear more consistent than fluctuating neural activity, best illustrated by occular artifacts, and can therefore be easily identified via ICA.
# 
# **An ICA is performed in multiply steps:**
# 
# 1. Specify the ICA algorithm/paramertes:
#     - n_components = the number of components we want to decompose our signal in. The more components one includes the more accurate the ICA solution becomes, i.e. the finer grained your sources of variability will be identified in the specific components.
#     - max_iter = the number of iterations for approximation (see <a href="https://en.wikipedia.org/wiki/Iterative_method" title="https://en.wikipedia.org/wiki/Iterative_method">Wikipedia: iterative processes</a> for more)
#     - random_state = identifier for the initial value/decomposition matrix used (as ICA is non-deterministic this is necessary to reproduce the same output when repeating the ICA)
#     - method = the ICA algorithm to use, defaults to "fastica". Choice of algorithm usually boils down to computational speed vs. maximization of information (see the <a href="https://mne.tools/stable/generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA" title="https://mne.tools/stable/generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA">MNE API: preprocessing.ICA</a> for more  information)
#     
# 
# 2. Fit the ICA to the data/Identify ICA components
#     - visualize your component solution, i.e artifact detection
#     - for an introductionary tutorials to identifying ICA components in eeg see <a href="https://labeling.ucsd.edu/tutorial/labels" title="ICLabel Tutorial: EEG Independent Component Labeling">ICLabel Tutorial: EEG Independent Component Labeling </a>
# 
# 3. Specify which components should be removed from the data
# 
# 4. Reconstructing the Signal, i.e. apply the ICA solution with the components containing artifacts excluded from the data
# <br/>
# 
# **Further reading**
# 
# For more theory on the ICA and how it's used in MNE etc. see the <a href="https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html" title="MNE tutorial: artifact correction via ica"> MNE tutorial: artifact correction via ica </a> and  <a href="https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html#sphx-glr-auto-examples-decomposition-plot-ica-blind-source-separation-py" title="Scikit-learn example: ica blind source separation">Scikit-learn example: ica blind source separation </a>
# 
# For more general explanations on ICA for EEG data, what to look for, which artifacts to include etc., see the video <a href="https://www.youtube.com/watch?v=kWAjhXr7pT4&list=PLXc9qfVbMMN2uDadxZ_OEsHjzcRtlLNxc" title="Youtube: Independent components analysis for removing artifacts by Mike X Cohen"> Independent components analysis for removing artifacts by Mike X Cohen</a> or the video series on <a href="https://www.youtube.com/watch?v=kWAjhXr7pT4&list=PLXc9qfVbMMN2uDadxZ_OEsHjzcRtlLNxc" title="Youtube: ICA applied to EEG time series by Arnaud Delorme"> ICA applied to EEG time series by Arnaud Delorme</a>.
# 
# For this tutorial we'll be excluding ICA components by hand (or eye), but to remove researcher bias and allow for analysis of larger sample sizes we could also employ ica templates based on certain common artifacts and automate this process. See <a href="https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html#selecting-ica-components-using-template-matching" title="MNE tutorial: selecting ica components using template matching">MNE tutorial: selecting ica components using template matching </a> for more info.
# 

# <br/>
# 
# **1. Specify the ICA algorithm/paramertes:**

# In[35]:


ica = mne.preprocessing.ICA(n_components=20,
                            random_state=7,
                            max_iter='auto', 
                            method='infomax')  # set-up ICA parameters


# <br/>
# 
# **2. Fit the ICA to the data/Identify ICA components**

# In[36]:


ICA_solution = ica.fit(raw_filtered)  # fit the ICA to our data


# <br/>
# 
# **Now we have a number of ways for visualizing the identified sources of variability.**
# 
# We can plot the ICA componets as a projection on the sensor topography, to see where the variability in our signal is located in sensor space. E.g. occular activiy should be located over frontal electrodes, like in component ICA 000 and component ICA 003

# In[37]:


ica.plot_components(picks=range(0, 20));  # visualize our components


# <br/>
# 
# Another way to visualize our ICA solution and one of the easiest ways to understand what patterns of variability in our data an ICA component captures, is to visualize the components in source space, i.e. how do the components represent latent sources of signal variability over time.
# See <a href="https://mne.tools/stable/generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA.plot_sources" title="MNE API: plot_sources()">MNE API: plot_sources() </a> for more info.

# In[38]:


ica.plot_sources(raw_filtered);


# <br/>
# You can scroll through the plot_sources() output (if opened via binder or jupyter notebooks/lab) and compare it with the topographical map above, to see if you can spot some more artifcats or if you can identify genuine sources of neural activity.
# 
# Here we see that the first component (ICA 000) appears to capture our blink artifacts, while the later parts of ICA 003 appears to be indicative of horizontal eye movement. The topographical location of these artifacts in the plot above points towards the same intepretation. ICA 11 and ICA 18 additionaly appear to capture some weird artifacts, possibly due to channel noise or electrical interference.

# To visualize what would happen to our signal if we were to extract these components, we can use the  <a href="https://mne.tools/stable/generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA.plot_overlay" title="MNE API: plot_overlay()">plot_overlay() </a> function and pass the problematic ica components by their index as arguments.

# In[39]:


ica.plot_overlay(raw_filtered, exclude=[0, 3]);


# <br/>
# 
# And it appears that the exclusion of the two components results in the elimination of the blink artifacts.

# _Note of caution: It's generally advised not to exclude components if you're not certain that they may contain artifacts. Adjustments for artifcats that are only present in a small subset of all events (e.g. presented stimuli) can still be made later by excluding the affected epochs. So we'll stick with the exclusion of these 2 components for now._

# <br/>
# 
# **3. Specify which components you want to remove from the data**
# 
# We'll add the components we want to exclude from our data to the ica.exlude object (which can be done the same way as we did above withe raw.info[bads] list), create a copy of our filtered data (as to not modify it in place) and apply the ica solution.

# In[40]:


# apply ica to data
ica.exclude = [0, 3]


# <br/>
# 
# **4. Reconstructing the Signal**

# In[41]:


reconst_sig = raw_filtered.copy().pick_types(eeg=True, exclude='bads')
ica.apply(reconst_sig)


# <br/>
# 
# **And lastly comparing our ICA reconstructed signal to the filtered data, it seems that we got rid of most ocular artifacts.**

# In[42]:


raw_filtered.plot();


# In[43]:


reconst_sig.plot();


# <br/>
# 
# To free up space we get rid of the filtered data

# In[44]:


del raw_filtered


# <br/>
# 
# **Now we can save the preprocessed data for further processing.**
# 
# Let's first get our paths in order.

# In[45]:


homedir = os.path.expanduser("~")  # find home diretory
output_path = (homedir + str(os.sep) + 'msc_05_example_bids_dataset' + 
               str(os.sep) + 'sub-test' + str(os.sep) + 'ses-001' + str(os.sep) + 'meg')
                         

print(output_path)


# <br/>
# 
# and save again using the  <a href="https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.save" title="MNE API: mne.raw.save()"> raw.save() </a> function.**

# In[46]:


reconst_sig.save(output_path + ((os.sep) +
                                'sub-test_ses-001_sample_audiovis_eeg_data_preprocessed.fif'),  # the filename
         picks=['eeg', 'eog', 'stim'],  # the channels to pick (could also be single channels)
         overwrite=True)  # checks if data is already present and overwrites if true


# <br/>
# 
# **Other techniques for artifact detection/repair?**

# For an overview of other techniques for preprocessing you can follow these MNE tutorials:
# 
# <a href="https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html" title="https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html">MNE tutorial: Repairing artifacts with Signal-space projection (SSP)</a>
# 
# <a href="https://mne.tools/stable/auto_tutorials/preprocessing/60_maxwell_filtering_sss.html" title="https://mne.tools/stable/auto_tutorials/preprocessing/60_maxwell_filtering_sss.html">MNE tutorial: Signal-space separation (SSS) and Maxwell filtering</a>
# 
# <a href="https://www.youtube.com/watch?v=6njzcZWuh9Q&list=PLn0OLiymPak2gDD-VDA90w9_iGDgOOb2o" title="https://www.youtube.com/watch?v=6njzcZWuh9Q&list=PLn0OLiymPak2gDD-VDA90w9_iGDgOOb2o">MNE tutorial: Surface Laplacian for cleaning, topoplogical localization, and connectivity</a>

# <br/>
# <br/>
# 
# ## 4. Epoching

# ### Epochs
# 
# Epochs represent **chunks of continous eeg data time-locked to certain events or periods of activity** (e.g. experimental blocks). They are usually equal length and should contain enough time before an  event to allow for baseline correction and enough time after the stimulus to catch all possible activity (evoked/event-related potentials or changes in frequency band) of interest for later analysis.
# 
# 
# The mne.Epochs objects can be treated the same as the previous raw objects in a number of ways, including the .info() manipulations we
# 
# **To create our epochs we  need two things:**
#     - raw or preprocessed data
#     - events
#     
# For more info on the Epochs data structure see the <a href="https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html#sphx-glr-auto-tutorials-epochs-10-epochs-overview-py" title="MNE tutorial: The Epochs data structure: discontinuous data)">MNE tutorial: The Epochs data structure: discontinuous data</a>

# ### Events
# 
# Events in the context of this tutorial are **flags indicating that at a certain time the presentation of a stimulus or a participant's reaction has occured**. Event IDs indicate what kind of condition was recorded, usually they are stored as simple integers. 
# 
# Information on what kind of stimulus/reaction was recorded is in general stored in the "Stim" channels or the metadata of the raw object (check raw.info['events']). Otherwise, if the Data is organized in the BIDS format, you can check the "_events.tsv" file in the subject folder and for the possible descriptions of the contained event-identifiers the "dataset_description.json" or the "README.txt". Some systems save events in a separate annotation file (e.g. eeglab set files), for which mne provides a dedicated function to convert these files to an event array. See <a href="https://mne.tools/stable/auto_tutorials/intro/20_events_from_raw.html" title="https://mne.tools/stable/auto_tutorials/intro/20_events_from_raw.html">MNE tutorial: Parsing events from raw data</a> for more info.

# In[47]:


reconst_sig.info['events']  # check if info on events is contained in raw object


# In[48]:


raw.info['ch_names']  # see if stim_channel are found in the raw object


# <br/>
# 
# ### 4.1 finding/renaming events

# To find the events in the raw file we can usually use <a href="https://mne.tools/stable/generated/mne.find_events.html">find_events()</a> function on our raw object. This function automatically checks if there are Stim channel present in the data. Stimulus channel do not collect EEG data, but the triggers associated with events that an experiment may contain.

# In[49]:


events = mne.find_events(raw)


# <br/>
# 
# We ca visualize these stimulus channel using the <a href="https://mne.tools/stable/generated/mne.pick_types.html" title="https://mne.tools/stable/generated/mne.pick_types.html">.pick_types()</a> function on our raw when we sett the "stim" parameter to true.

# In[50]:


raw.copy().pick_types(meg=False, stim=True).plot();  # copy as to not modify our data in place


# <br/>
# 
# Looking at these channels it becomes obvious that "STI 014" contains all relevant events contained in the other stimuli channel and encodes them by variations in amplitude. Therefore explicitly picking the events from this channel leads to the same result as the function above.
# 
# Looking at the other stim channels reveals that they each contain only a single kind of event. E.g. "STI 002" contains only the event with the ID "5"

# In[51]:


events = mne.find_events(raw, stim_channel='STI 002')


# <br/>
# 
# Let's get all events

# In[52]:


events = mne.find_events(raw, stim_channel='STI 014')


# <br/>
# 
# In the <a href="https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html" title="https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html">MNE: Epochs tutorial</a>, a list of labels for the events in the sample data corresponding to our event_id's is provided, which we can use to make our data a little more readable.

# In[53]:


event_id = {'Auditory/Left': 1, 'Auditory/Right': 2,
            'Visual/Left': 3, 'Visual/Right': 4,
            'smiley': 5, 'button': 32}


# <br/>
# 
# We can further use these descriptions when visulaizing our events using the <a href="https://mne.tools/stable/generated/mne.viz.plot_events.html" title="https://mne.tools/stable/generated/mne.viz.plot_events.html">plot_events()</a>
# function

# In[54]:


#Specify colors for events for the plot legend
color = {1: 'green', 2: 'yellow', 3: 'red', 4: 'c', 5: 'black', 32: 'blue'}

# Plot the events to get an idea of the paradigm
mne.viz.plot_events(events, 
                    raw.info['sfreq'],  # sampling frequency
                    raw.first_samp,  # starting point of plot (first sample)
                    color=color,  # the color dict specified above
                    event_id=event_id,  # our event dictionary
                    equal_spacing=True);


# <br/>
# 
# **With this information we can finally create our epochs object using the <a href="https://mne.tools/stable/generated/mne.Epochs.html" title="MNE API: mne.Epochs()">mne.Epochs()</a> constructor function**
# 
# 
# Baseline correction will be automatically applied given the intervall between tmin (epoch start) and t = 0 (i.e. the relevant event) of specified epochs, but can be specified by adding the baseline parameter to the mne.Epochs() constructor. Baseline correction  is applied to each channel and epoch respetively by calculating the mean for the baseline period and substracting this mean from the signal of the entire epoch.
# 

# In[55]:


# like with raw objects data is not automatically loaded into memory, therefore we set preload=True
epochs = mne.Epochs(reconst_sig, 
                    events, 
                    picks=['eeg'], 
                    tmin=-0.3,  # start of epoch relative to event in seconds
                    tmax=0.7, # end of epoch relative to event
                    #baseline=(-0.3,0),
                    preload=True)


# In[56]:


epochs.info


# In[57]:


print(epochs)


# <br/>
# Let's recreate our dictionary containing the explicit event names and add our event dict to the event object, so that we can later access data based on these labels 

# In[58]:


event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'buttonpress': 32}

epochs.event_id = event_dict
print(epochs)
print(epochs.event_id) 


# <br/>
# 
# **We can visualize epochs in the same way we visualized raw data previously**

# In[59]:


epochs.plot(n_epochs=5);  # pick first 5 epochs


# In[60]:


epochs.plot(events=events);  #or plot all by event type


# <br/>
# 
# **We can then select epochs based on the events they are centered around in a number of ways.**

# In[61]:


epochs.event_id # recall what our events are


# <br/>
# Select by a specific group of events (MNE treats the "/" character as a string separator, so we can explicitly call only auditory epochs etc.)

# In[62]:


epochs['auditory']  # epochs centered around presentation of auditory stimulus


# In[63]:


epochs['visual']  # epochs centered around presentation of visual stimulus


# In[64]:


epochs['right']  # subselect only stimuli presented on the right side


# <br/>
# select by multiple tags (i.e select all epochs containing either "right" or "auditory")

# In[65]:


epochs[['right', 'auditory']]


# <br/>
# or by using the explicit label of the specific event

# In[66]:


# using explicit label 
epochs[['auditory/right']]


# In[67]:


# using explicit label 
epochs[['auditory/left']]


# <br/>
# 
# **We can further select epochs by their index using list comprehensions.**

# In[68]:


print(epochs[:10])    # first 10 epochs
print(epochs[:10:2])  # epochs 1, 3, 5, 7, 9; i.e [start:stop:step-size]
print(epochs['auditory'][:10:2])  # or select explicitly
print(epochs['auditory'][[0, 2, 4, 6, 8]])  # or select explicitly

epochs['auditory'][[0, 2, 4, 6, 8]].plot();


# <br/>
# 
# and access the contained data again via the get_data() function, which returns a np.array of the shape (number of epochs, number of channels, number of time-points)

# In[69]:


print(epochs.get_data())  # array (n_epochs, n_channels, n_times)
epochs.get_data().shape


# <br/>
# 
# return data for all channels and all time-points for the first 2 epochs

# In[70]:


epochs.get_data()[:1]  # first 2 epochs


# In[71]:


# get data only from last channel of first epoch
epochs.get_data()[0][58] # array of shape (n_epochs, n_channels, n_times)


# <br/>
# 
# So to get get the signal amplitude for the last time_point only from the last channel of the first epoch, we would do the following

# In[72]:


epochs.get_data()[0][58][600] # array of shape (n_epochs, n_channels, n_times)


# <br/>
# 
# **At this point it honestly get's quite confusing, so it might be preferable to switch to a pandas DataFrame, although this may cost computation time later on if you're going to iterate over data for group level analysis etc.**

# In[73]:


df = epochs.to_data_frame() # create dataframe with nested index
df


# In[74]:


df['EEG 001']


# <br/>
# access DataFrame for the first 3 epochs

# In[75]:


df.loc[df['epoch'] == 2]  


# <br/>
# access epochs containing certain conditions

# In[76]:


df.loc[df['condition'] == 'auditory/right']  # access epochs containing certain condition


# <br/>
# looking at the first epoch for auditory stimuli presented on the right

# In[77]:


df.loc[(df['condition'] == 'auditory/right') & (df['epoch'] == 0)]


# <br/>
# looking at the signal from a single electrode for the first epoch for auditory stimuli presented on the right (jeez)

# In[78]:


# looking at a subset of the dataframe
df.loc[(df['condition'] == 'auditory/right') & (df['epoch'] == 0)]['EEG 001']


# <br/>
# 
# ### 4.2 rejecting bad epochs

# As discussed earlier we'll get rid of epochs containing artifacts.
# 
# This way we'll only reject a small subset of relevant data, while the rejection of ICA components or whole channels might have lead to a greater loss of relevant data in the signal. This is one of the reasons why we collect so many trials in EEG experiments. 
# 
# As a point of caution, epoch rejection is best done after repairing our signal for eye-blinks, as otherwise we might lose a significant number of epochs. 
# - for more information see the <a href="https://mne.tools/stable/auto_tutorials/preprocessing/20_rejecting_bad_data.html#tut-reject-epochs-section" title="MNE tutorial: Rejecting Epochs based on channel amplitude">MNE tutorial: Rejecting Epochs based on channel amplitude</a>
# 
# For now we'll simply **reject eppochs by maximum(minimum peak-to-peak signal value thresholds** (exceeding 100 µV, as it's highly unlikely that any eeg-signal may reach this amplitude or epochs with channel with signal below 1 µV to get rid of flat channels) using the <a href="https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.drop_bad" title="https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.drop_bad">epochs.drop_bad()</a> function.
# 
# 

# In[79]:


# define upper and lower threshold as dict(channel_type:criteria)
reject_criteria_upper = dict(eeg=100e-6)  # 100 µV
reject_criteria_flat = dict(eeg=1e-6)  # 1 µV


# In[80]:


epochs.drop_bad(reject=reject_criteria_upper, flat=reject_criteria_flat)


# <br/>
# 
# We can then further visulaize how many epochs were dropped for each channel.

# In[81]:


epochs.plot_drop_log();


# <br/>
# If no arguments are provided for the drop_bad() function, all epochs that were marked as bad would be dropped.

# <br/>
# 
# **Another strategy for rejecting epochs is to declare the above criteria while creating the epochs object**
# 
# We can specify where in the epoch to look for artifacts with the **reject_tmin** and **reject_tmax** parameters of  <a href="https://mne.tools/stable/generated/mne.Epochs.html" title="MNE API: mne.Epochs()">mne.Epochs()</a>. Reject_tmin = 0 would mean that we look for artifacts after the event, while reject_tmax = 0 would mean that we only look for artifacts in the timeframe before the event occurde. If not specified they default to the ends of the timeframe specified for the epochs.

# In[82]:


epochs = mne.Epochs(reconst_sig, 
                    events, 
                    picks=['eeg'], 
                    tmin=-0.3, tmax=0.7,
                    reject_tmin=-0.3,
                    reject_tmax=0, 
                    reject=reject_criteria_upper, 
                    flat=reject_criteria_flat, 
                    preload=True)


# <br/>
# 
# For the automatic rejection of epochs we can use the **reject_by_annotation parameter**. 
# This can be achieved by adding information about the timeframe of an artifact, break, experimental block etc. to an <a href="https://mne.tools/stable/generated/mne.Annotations.html#mne.Annotations" title="https://mne.tools/stable/generated/mne.Annotations.html#mne.Annotations">mne.Annotations</a>  object and using the <a href="https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.set_annotations" title="https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.set_annotations">.set_annotations()</a> function to add this to the .info object of our data.
# 
# So we'll define 3 hypothetical breaks in the next step by **specfifying their onset time, duration and adding a description**. If the description contains the string "bad" annotated data-spans will be automatically excluded from most analysis steps, including the creation of epochs.

# In[83]:


onset_breaks = [reconst_sig.first_time + 30, reconst_sig.first_time + 90, reconst_sig.first_time + 150]
durations = [10, 10, 10]
description_breaks = ['bad_break_1', 'bad_break_2', 'bad_break_3']


# <br/>
# 
# Now we create our Annotations object

# In[84]:


breaks = mne.Annotations(onset=onset_breaks,
                               duration=durations,
                               description=description_breaks,
                               orig_time=raw.info['meas_date'])


# <br/>
# 
# and set the annotations to our raw object.

# In[85]:


reconst_sig.set_annotations(reconst_sig.annotations + breaks)


# <br/>
# Let's check how this look in our interactive plots

# In[86]:


reconst_sig.plot();


# <br/>
# 
# **Now when we create our epochs object, any epoch contained in the designated "breaks" will be excluded.**

# In[87]:


epochs = mne.Epochs(reconst_sig, 
                    events, 
                    picks=['eeg'], 
                    tmin=-0.3, tmax=0.7,
                    #reject_tmin=-0.3,
                    #reject_tmax=0, 
                    reject=reject_criteria_upper, 
                    flat=reject_criteria_flat,
                    reject_by_annotation=True,
                    preload=True)

epochs.event_id = event_dict # rename our events


# In[88]:


epochs.event_id


# <br/>
# 
# and to visualize the dropped epochs.

# In[89]:


epochs.plot_drop_log();


# <br/>
# 
# **Now we could convert this data into a dataframe for further maipulation or export our epochs as an tsv file in long format for statistical analysis.**

# In[90]:


df = epochs.to_data_frame(index=['condition', 'epoch', 'time']) # create dataframe with nested index
df.sort_index(inplace=True)
df


# In[91]:


df.to_csv(output_path + str(os.sep) + 'sub-test_ses-001_sample_audiovis_eeg_data_epo.tsv', 
          sep='\t')


# In[92]:


epochs.event_id


# <br/>
# 
# **Or save the epochs as a .fif file using epochs<a href="https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.save">.save()</a> which can be imported back into mne using the <a href="https://mne.tools/stable/generated/mne.read_epochs.html">.read_epochs()</a> function.**

# In[93]:


epochs.save(output_path + str(os.sep) + 'sub-test_ses-001_sample_audiovis_eeg_data_epo.fif',
            overwrite=True)

