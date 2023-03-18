#!/usr/bin/env python
# coding: utf-8

# # Electroencephalography
# 
# <br/>

# 
# **Simply put, Electroencephalography is a (usually) non-invasive method for measuring "brain activity".**
# 
# It is a commonly used tool in the neurosciences to establish links between variables of interest such as behavior/cognition/sensory perception/mental processing/differences in personality or psychopathology and their underlying neural mechanims or changes in brain states/traits. An introduction into why and how this is done will be establishd in this chapter.
# 
# Common non-research applications that may be of interest are:
# 
# - Clinical (localization of tumors, diagnosis, brain damage due to head injuries, epilepsy, encephalopathy, encephalitis, stroke, sleeping dsiorder; depth of anesthesia, and possible confirmation of brain death in coma patients)
# - Brain-computer interfaces (BCI) 
#     - <a href="https://www.frontiersin.org/articles/10.3389/fnhum.2018.00014/full" title="https://www.frontiersin.org/articles/10.3389/fnhum.2018.00014/full">EEG-Based Brain–Computer Interfaces for Communication and Rehabilitation of People with Motor Impairment</a>
#     
#     - Real important stuff:
#         - <a href="https://i2nk.co/mindwave-cat-ears" title="https://i2nk.co/mindwave-cat-ears">EEG Cat Ears</a>
#         - <a href="https://www.youtube.com/watch?v=-y8CHWx6aGY" title="https://www.youtube.com/watch?v=-y8CHWx6aGY">Mind Controlled Beer Pong </a>
# 
# <br/>
# If the language used is confusing head over to the
# <a href="https://link.springer.com/content/pdf/bbm%3A978-3-030-04573-9%2F1.pdf" title="https://link.springer.com/content/pdf/bbm%3A978-3-030-04573-9%2F1.pdf">EEG glossary</a>
# or for the more programming related stuff in the later chapters the
# <a href="https://mne.tools/stable/glossary.html" title="https://mne.tools/stable/glossary.html">MNE glossary </a>
# 
# 
# <br/>
# <br/>

# <br/>
# 
# ## Chapter overview:
#     
# **1. EEG in neuroscience research**
# - What's the use?
# - main variables of interest
# 
# **2. Biophysics**
# - What does "brain activity" or "signal" mean in the context of EEG?
# - Where does the EEG signal originate from?
#     - EEG source localization
# - What do we actually measure? What is the EEG code?
# - Neural oscillations as an integral functional mechanism of the brain
# - Frequency bands
# - Function of frequency bands
# 
# **3. Implications for EEG resarch**
# - What questions can we answer using EEG?
# - Conducting an EEG study in the context of cognitive neuroscience: a loose framework
#     1. Planning/idea to hypothesis
#     2. Study design
#     3. Study implementation
# 
# **4. Resources**
# - Tools for EEG-analysis
# - Where to obtain EEG data ?
# 
# <br/>

# ## Introduction to Electroencephalography

# 
# 
# ## 1. EEG in research/neuroscience

# **What's the use?**
# 
# The **main selling point of EEG compared to other methods is it's cost-effectiveness (when compared to MEG or fMRI) and portability (limited), combined with high temporal precision**. 
# Allowing for sampling rates of 1024Hz (one sample every 0.001s) and above. As neocortical fast-spiking neurons (association cortex) produce action potentials at a maximal mean frequency of 338 Hz {cite:p}`wang_firing_2016`, it should accordingly be possible to map even the fastest changes in neuronaly activity using EEG.
# 
# 
# Therefore EEG allows us to obersve the time course of cognitive processes in high resolution, so that we can divide a processing stream into **early components** _(usually linked to lower level perception (i.e. evoked potentials P100 (visual), N100 (unpredictable stimulus, multiple sensory modalities, P200 (visual search, word expectancy)_ and **later components** _(Event-related potentials; which are (usually) more  closely linked to conscious cognitive processing, such as evaluations or decision making)_ {cite:p}`beres_time_2017`. 
# 
# 
# We are further not limited to the observation of evoked or event-related potentials (i.e. the averaged signal following a task-demand or stimuli presentation), as the high temporal precision  allows us further to **identify neural oscillations**, the "rhythms of the brain", and how these relate to fluctuations of internal states or changes in external reality.
# 
# Therefore we can, with little investment, capture **rich, versatile data** (such as event-related & evoked activity, respective their source localization, changes in frequency & time-frequency content, functional connectivity and synchronicity/coherence between neural populations).

# 
# **The main variables of interest are EEG featuers, most commonly:**
# 
# 
# **Evoked potentials:**
# A wave or complex of waves in the contionously measured EEG signal "elicited by and time-locked to a physiological or nonphysiological stimulus or event" {cite:p}`mecarelli_clinical_2019`
# 
# 
# :::{figure-md} markdown-fig
# <img src="https://mne.tools/stable/_images/sphx_glr_20_visualize_evoked_008.png" alt="p100" class="bg-primary mb-1" width="600px">
# 
# Evoked potential (p100) after presentation of a visual stimulus on the right side of an participants field of view <a href="https://mne.tools/stable/auto_tutorials/evoked/20_visualize_evoked.html" title="https://mne.tools/stable/auto_tutorials/evoked/20_visualize_evoked.html">(MNE tutorials, visualizing evoked potentials, 2022)
# :::
#    
# 
# <br/>
# 
# **Event-related potentials:**
#     
# Same as evoked potentials, but usually show longer latency responses associated with events, as they are not directly thought to be related to basic sensory processing but to "higher" cognitive tasks such as decision making, response anticipation, evaluation of stimulus features (such as stimulus
# salience, template matching, novelty detection) {cite:p}`mecarelli_clinical_2019`.
# 
# 
# For an overview of different ERPs see e.g. <a href="https://www.neuroelectrics.com/blog/2014/12/18/14-event-related-potentials-erp-components-and-modaliaties/" title="https://www.neuroelectrics.com/blog/2014/12/18/14-event-related-potentials-erp-components-and-modalities/">Event Related Potentials (ERP) – Components and Modalities</a>.
# 
# 
# 
# 
# ```{figure} ../../../static/neuroscience/ERN-incongruent_flanker.png
# ---
# width: 550px
# name: ERN
# ---
# Event-related-potential: Event-related negativity over central electrodes ~ 50 ms (~ 4µV over FCz) following error commission in a flanker task, 0 is time-locked to a motor response (Master Thesis, Michael Ernst, 2021).
# ```
# 
# <br/>
# 
# **Neural oscillations:**
# 
# Repetitive, rhythmic patterns in the synchronous activity of neurons or neuron clusters that can be described according to their frequency, phase and amplitude. Usually divided into frequency bands alpha, beta, gamma and so on <a href="http://learn.neurotechedu.com/neural%20oscillations/" title="http://learn.neurotechedu.com/neural%20oscillations/">(NEUROTECHEDU: What are Neural Oscillations?)</a>.
# 
# <br/>
# <br/>
# <br/>
# 
# 
# :::{figure-md} markdown-fig
# <img src="https://raphaelvallat.com/images/tutorials/bandpower/brain_waves.png" alt="neural oscillations in time domain" class="bg-primary mb-1" width="600px">
# 
# Illustration of Alpha, Beta, Gamma and Delta frequency bands <br/> <a href="https://raphaelvallat.com/bandpower.html" title="https://raphaelvallat.com/bandpower.html">(Compute the average bandpower of an EEG signal; Raphael Vallat, 2018)</a>
# :::
# 
# 
# 
# 

# <br/>
# 
# ## 2. Biophysics
# 
# **What does "brain activity" or "signal" mean in the context of EEG?**
# 
# 
# A simple answer would be that we are measuring the **summed activity of a number of postsynaptic potentials (i.e the exchange of electrochemical signals) at the scalp**. 
# 
# As the activity of a single neuron or the change in potential at a singluar postynaptic site does not elicit a change in voltage strong enough to be measured at the scalp, **only the summation of electrical potentials generated by number of synchronously active neurons that show similar spatial orientation** (otherwise their signals cancel each other out) prodcues a signal large enough to be detected at the scalp. This is illustrated in the following graphs, taken from a great resource to understand the biophysics behind the origin of the EEG signal:  {cite:p}`jackson_neurophysiological_2014`
# 
# 
# 
# :::{figure-md} markdown-fig
# <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSc9nk1vEJfRgIT_jx6BdXkcR_frSq1lWxWIRgYXg3u98Kj6ydEDa8o4-Y3b5qSpH4SDGU&usqp=CAU" alt="differences in amplitude signage" class="bg-primary mb-1" width="200px">
# 
# Illustration of differences in amplitude signage at the scalp electrode, <br/> dependent ob the spatial orientation of the dipole of a neuron {cite:p}`jackson_neurophysiological_2014`
# :::
# 
# 
# :::{figure-md} markdown-fig
# <img src="https://www.researchgate.net/profile/Yves-Matanga/publication/326175998/figure/fig7/AS:644582983364616@1530691967168/EEG-signal-origin-from-neurons-to-electrodes-Neurons-activation-is-synchronized-and.png" alt="Synchronized activity of spatialy similarly oriented neural populations mesaured at the scalp electrode" class="bg-primary mb-1" width="350px">
# 
# Synchronized activity of spatialy similarly oriented neural populations mesaured at the scalp electrode {cite:p}`jackson_neurophysiological_2014`
# :::
# 
# 
# 
# 
# **A more complex, neuroanatomically informed answer would encompass:**
# 
# EEG is a method for the (continuous) measurement of spatiotemporally smoothed field potentials above a specific region of the scalp, i.e. the "extracellular currents reflecting the summed dendritic postsynaptic potentials in thousands to millions of pyramidal cells in parallel alignment" (Cf. {cite:p}`cohen_where_2017`) measured at an electrode X placed above region/coordinate Y integrated over an area of 10cm² or more (Cf. {cite:p}`buzsaki_origin_2012`).
# 

# 
# 
# ### Where does the EEG signal originate from?
# 
# 
# **EEG source localization**
# 
# One disadavantage of the EEG is that it is difficult to estimate where neuronal activity originates from, given only the signal measured at the scalp, as the **spatial resolution of EEG is fairly limited** (centimeter range).
# 
# The spatial resolution of the EEG is unfortunately not only limted by the number of electrodes we can place on willing participants (commonly 32-64 channel systems for research; up to 192 channel systems exist) but **dependendent on the propagation of the current flow though multiple membranes and the skull**, which due to it's low electrical conductivity and homogenous thickness leads to local changes in conductivity, as well as the skulls non-spherical (and individually varying non-uniform) shape, which results in differences in the distance between electrodes and the center of the head (or varying regions of interest) {cite:p}`michel_eeg_2019`.
# 
# 
# **To find out where the signal measured at an electrode originates, we therfore have to solve a number of problems.**
# 
# **1.** How to estimate the signal at a given electrode from a given source, which is understood as the **forward problem** of EEG. 
# 
# The forward problem can be **overcome with the standardization of tools and methods of measurement**. E.g. the standardized 10-20 system that not only describes electrode position and nomenclature, but includes instructions on how to measure a participant's skulls (Skull circumference, distance preauricular to preauricular and distance Nasion to Inion) and position the EEG-cap. For estimated differences in conductivity due to varying skull thickness ideally an MRI scan of the subject would be provided.
# 
# We'll have to solve the forward problem, before turining towards what's actually of interest when it comes to source localization of EEG features, which is the ... 
# 
# **2.** **Inverse problem**, i.e. which source in the brain produces the signal observed at a given electrode. 
# 
# Unfortunately the same signal measured at the scalp by a given electrode can be produced by a number of non-unique source distributions, as e.g. the activity (current) generated from different neuron pools may cancel each other out dependent on their relative positions {cite:p}`michel_eeg_2019`. **Meaning that we're dependent on mathematical models (such as LORETA {cite:p}`pascual-marqui_low_1994`) that include a priori assumptions on how the electrical (field) potential produced by neuronal activity propagates through different layers of tissue and bone to reach a given electrode** {cite:p}`michel_eeg_2019`. This is generally done by dividing the whole brain volume into a 3d grid of volumetric solution points (voxels) and calculating how a specific pattern of current density (electric current per cross-sectional area) would have to be distributed to be producde by a given source, i.e. a voxel space.
# 
# 
# For a detailed explanation and tutorial of EEG source-estimation see the <a href="https://neuroimage.usc.edu/brainstorm/Tutorials/SourceEstimation" title="https://neuroimage.usc.edu/brainstorm/Tutorials/SourceEstimation">Brainstorm tutorial: source estimation</a>
# 
# 

# ### What do we actually measure? What is the EEG code?
# 
# 
# **Now, how do we get from the summed activity of neurons to complex, measurable features that are cause or correlation of cognitive and behavioral processes**? 
# 
# <br/>
# Unfortunately, that is above my pay-grade to answer, honestly.<img style="float:right;;width:225px; height:165px; bottom:0px; border:none;" src="https://i.gifer.com/MHS.gif" alt="https://i.gifer.com/MHS.gif">
# 
# <br/>
# 
# {cite:p}`cohen_where_2017` describes the question of what the EEG signal is, as basically the _**"The Ultimate but Ultimately Unattainable Goal: One-to-One Mapping between EEG Feature and Microcircuit Conﬁguration"**_, but offers some insight on why we might justiﬁably  be optimistic that this seemingly unattaiable goal will eventually be achieved. Head over to the linked paper if you want to know more.
# 
# 
# <br/>
# 
# <img src="https://media3.giphy.com/media/pPhyAv5t9V8djyRFJH/giphy.gif?cid=790b7611594b6670cf91529ee4084ea301f013771e91dc85&rid=giphy.gif&ct=g" alt="https://media3.giphy.com/media/pPhyAv5t9V8djyRFJH/giphy.gif?cid=790b7611594b6670cf91529ee4084ea301f013771e91dc85&rid=giphy.gif&ct=g" style="float:left; margin-right:10px;width:160px;height:auto;" />
# Apparently, (on an ultimately rational level) most researches are content with observing correlations of ERPs/Evoked potentials/changes in neural oscillations and behavioural aspects/presented stimuli.
# 
# 
# 

# 
# <br/>
# <br/>
# <br/>
# 
# But the underlying implications for the EEG may on the other hand be explored when we view the EEG as not merely a tool to visualize the isolated activity of a neuron population that produces a "spike" of activity for a temporally distinct event and instead focus on the inherent rhytmic activity of the brain.
# 
# ### Neural oscillations as an integral functional mechanism of the brain
# 
# 
# ***"Brains are foretelling devices and their predictive powers emerge from the various rhythms they perpetually generate. At the same time, brain activity can be tuned to become an ideal observer of the environment, due to an organized system of rhythms. The speciﬁc physiological functions of brain rhythms vary from the obvious to the utterly impenetrable"*** (p.8, {cite:p}`buzsaki_rhythms_2006`)
# 
# 
# What we can take from this is that:
# 
# **a.** **The brain is not a static organ, it's function is not limited to the mere processing of our outside world, but instead the prediction of possible external events** (see also {cite:p}`friston_history_2012`).
# 
# 
# **b.** **Rhythms are necessary for the predictive nature of our conscious existence**
# 
# - (such as the phase of oscillations in the alpha band that is predictive of (temporal) perceptual outcomes in visual discrimination tasks and shows significant phase shifts towards an individuals optimal phase for stimulus discrimination when presented with predictive cues about the temporal nature of a future event {cite:p}`samaha_top-down_2015`.
# 
# 
# **c.** **Rhythms can "tune" the brain towards different "states" of activity** (e.g. when switching from attentive to relaxed to different stages of sleep)
# 
# 
# It follows that the brain can be understood as a **system of functionally connected oscillators (or pattern generators) and that the perturbations of the generated rhythms due to internal or external factors are the main mechanisms that allow us to adapt to our surroundings**. They are not only relevant for prediction, but also the synchronization of operations/activity between local neuron clusters and more distant regions. Functions without which we'd suffer from a plethora of "rhythm-based cognitive maladies", such as epilepsy, Parkinson's or sleep disorders (p.9, {cite:p}`buzsaki_rhythms_2006`).
# 
# 
# One of the simplest forms that an oscillator may take and therefore a good illustration for the usefulness of rhythmic activity is the **central pattern generator (CGP)**. Central pattern generators are used for walking, breathing and other rhythmical behavioral patterns (p.144, {cite:p}`buzsaki_rhythms_2006`). The article <a href="https://www.audradavidson.com/post/if-i-only-had-a-spine" title="https://www.audradavidson.com/post/if-i-only-had-a-spine">"If i only had a spine" by Audra Davidson</a>, that the illustration below is borrowed from, shows a very simplistic CGP, containing only 2 neurons at the level of the spinal cord and given a “go” signal from the brain, allows for the production of ongoing synchronized activity according to a pre-specified plan, i.e. extending and contracting a target muscle, until the brain sends further instructions. 
# 
# 
# :::{figure-md} markdown-fig
# <img src="https://static.wixstatic.com/media/99c553_8518f036bd53407abf89746f757fb19a~mv2_d_2100_1500_s_2.gif" alt="Audra Davidson (2019) if-i-only-had-a-spine" class="bg-primary mb-1" width="500px">
# 
# Illustration of a central pattern generator (<a href="https://www.pnas.org/doi/abs/10.1073/pnas.1503686112" title="https://www.pnas.org/doi/abs/10.1073/pnas.1503686112"> Audra Davidson (2019)</a>).
# :::
# 
# <br/>
# 
# If we now further complicate this pattern of synchronous exictatory activations aimed a different target sites by introducing inhibitory neurons, rather quickly non-linearity in the organized activity of cortical circuits emerges. While the fine balance between excitatory and inhibitiory activations is often accomplished by oscillations, with the introduction of non-linearity, chaos (theory) follows (p.62, {cite:p}`buzsaki_rhythms_2006`). The emerging perspective is that **the brain constitutes a system that at most times operates in a state of "self-organized criticality", or put differently the "border between predictable periodic behavior and unpredictable chaos"** (p.145, {cite:p}`buzsaki_rhythms_2006`) While this may sound problematic at first glance, it is a necessary attribute of our nervous system, as **such a state allows for dynamics that favor quick and flexible reactions to internal and external perturbations**. Therefore this critical state poses a _"clear advantage for the cerebral cortex since it can respond and reorganize it's dynamics in response to the smallest and weakest perturbations"_ (p.145, {cite:p}`buzsaki_rhythms_2006`).
# Once we then react to an external perturbation or direct our attention towards a certain task, we may switch to robust, predictable oscillatory activity (p.111, {cite:p}`buzsaki_rhythms_2006`). 
# 
# 
# ### Frequency bands
# 
# These **states of oscillatory synchrony are believed to be constrained to certain patterns or "frequency bands" that give rise to a number of different functions**. In EEG literature you'll generally come across the following frequency bands, although their exact borders are bound to slightly differ between publications, as they were originally drawn somewhat arbitrarily out of the necessity to accomodate early EEG recording technology.
# <br/>
# 
#                                         delta - 0.5–4 hertz
#                                         theta - 4–8 hertz
#                                         alpha - 8–12 hertz
#                                         beta - 12–30 hertz
#                                         gamma - > 30 hertz
# <br/>
# 
# 
# Other definitions that are closer to practictal considerations exist, e.g. <a href="http://www.medicine.mcgill.ca/physio/vlab/biomed_signals/eeg_n.htm" title="http://www.medicine.mcgill.ca/physio/vlab/biomed_signals/eeg_n.htm">provided by the McGill Physiology Lab for Biomedical Signals Acquisition</a>, which additionally offers descriptions of commonly observed amplitude and location.  
# 
# Or simply head over to <a href="https://en.wikipedia.org/wiki/Electroencephalography#Comparison%20of%20EEG%20bands" title="https://en.wikipedia.org/wiki/Electroencephalography#Comparison%20of%20EEG%20bands">Wikipedia; EEG Comparison of frequency bands  </a> 
# <img style="float:middle;margin-left:10px;width:125px; bottom:0px; border:none;" src="https://thumbs.gfycat.com/NarrowSafeAmericanpainthorse-size_restricted.gif" alt="It's_fine.gif">
# 
# 
# <br/>
# <br/>
# 
# _In general it should be noted that it is good practice to define your frequency range of interest when searching or publishing studies, as the important info is the actual frequency band used not the potential buzzword._
# 

# <br/>
# 
# ### Function of frequency bands
# 
# Frequency bands can be understood to define the gates of temporal processing, as each oscillatory cylce constitutes a temporary processing window, which (may) signal the beginning and ending of an encoded piece of neuronal information (p.115, {cite:p}`buzsaki_rhythms_2006`).
# 
# 
# The frequency of an oscillatory band determines the length of this temporal window and is in turn (partly) determined by the size of the neuronal pool that produces the frequency band. While **slow oscillatons usually involve greater numbers of neurons over larger areas of the brain, fast oscillations are used for local integration requiring short time windows** (p.116, {cite:p}`buzsaki_rhythms_2006`). For example, alpha and theta rhythms are thought to bind more distant areas into functional units, thereby modulating the activity of locally more restrained clusters controlled by faster oscillatory activity {cite:p}`leuchter_resting-state_2012)`.
# 
# 
# 
# As there is simply no way to write a comprehensive review of associated functions and regional differences for all commonly observed frequency bands given the format of this course, below we'll list some sample literature on modern interpretations/research on the possible functions that the most common frequency bands are associated with.
# 
# 
# **Delta**
# 
# - <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2635577/" title="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2635577/">Sleep classification according to AASM and Rechtschaffen & Kales: effects on sleep scoring parameters  {cite:p}`moser_sleep_2009`</a>
# 
# 
# 
# - <a href="https://www.sciencedirect.com/science/article/abs/pii/S0149763411001849?via%3Dihub" title="https://www.sciencedirect.com/science/article/abs/pii/S0149763411001849?via%3Dihub">EEG delta oscillations as a correlate of basic homeostatic and motivational processes {cite:p}`knyazev_eeg_2012`</a>
# 
# 
# 
# **Gamma**
# 
# - Gamma rhythms mediate feedforward communication <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2635577/" title="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2635577/">(Cortical layers, rhythms and BOLD signals {cite:p}`scheeringa_cortical_2019`)</a>
# 
# 
# **Theta**
# 
# - <a href="https://www.sciencedirect.com/science/article/pii/S1364661314001077?casa_token=s0m3fYGxtcAAAAAA:0HbWQYs6e-7pBUV4AvrQ0lhXGj065Pf65iy3hnIOxVlNZt4XhLx9-W1HlxgcWDowQfQ0Ez6NVBk" title="https://www.sciencedirect.com/science/article/pii/S1364661314001077?casa_token=s0m3fYGxtcAAAAAA:0HbWQYs6e-7pBUV4AvrQ0lhXGj065Pf65iy3hnIOxVlNZt4XhLx9-W1HlxgcWDowQfQ0Ez6NVBk">Frontal theta as a mechanism for cognitive control {cite:p}`cavanagh_frontal_2014-1`</a>
# 
# 
# **Alpha**
# 
# - <a href="https://onlinelibrary.wiley.com/doi/abs/10.1111/ejn.13747" title="https://onlinelibrary.wiley.com/doi/abs/10.1111/ejn.13747">The many characters of visual alpha oscillations {cite:p}`clayton_many_2018`</a>
# 
# 
# 
# **Beta**
# 
# - <a href="https://www.frontiersin.org/articles/10.3389/fnsys.2021.655886/" title="https://www.frontiersin.org/articles/10.3389/fnsys.2021.655886/">Understanding the Role of Sensorimotor Beta Oscillations {cite:p}`barone_understanding_2021`</a>
# 
# <br/>
# 
# If you want to know more about how rhythmic activity defines the neural system or how rhythms emerge, i highly advise the book <a href="https://neurophysics.ucsd.edu/courses/physics_171/Buzsaki%20G.%20Rhythms%20of%20the%20brain.pdf" title="https://neurophysics.ucsd.edu/courses/physics_171/Buzsaki%20G.%20Rhythms%20of%20the%20brain.pdf">"Rhythms of the Brain" by Buzsáki (2006)</a> for further reading.
# 
# <br/>

# ## 3. Implications for EEG resarch

# ### What questions can we answer using EEG?
# 
# **How and according to which rules does the brain react to external/internal perturbations and how/why does it generate internal perturbations?**
# 
# 
# - How **encoding** takes place (ERP; Change in power/coherence/phase-perturbation in a specfific frequency band; cross-frequency coupling etc.)
# 
# - Does an EEG component reflect a certain **function**? **Content** that the EEG feature represents? (e.g. algorithms for the integration of information, prediction, preparation of an action)
# 
# - **Timing or timeline of neural processes**? 
#     - How does information travel across neural populations? I.e. what's the flow of "neural" information when e.g. a stimuli was presented?
# 
# - In what **areas** does "computation" takes place? (**Source localization**; usually better answered with fMRI)
# 
# 
# **So we can**
# - map brain activity to behaviour/external factors
# - explore the (possible mathematical/methodological) laws according to which information is received, transformed or applied
# - and find which areas are involved/responsible

# 
# _An aspect that we'll not explore further, that is especially prominent in **clinical and personality psychology**: Do Indviduals differ in EEG features based on certain traits/staes (e.g. psychopathology)?_
# 
# Two approaches to answer questions of this nature are exemplified in the 2 studies below:
# 
# <a href="https://www.frontiersin.org/articles/10.3389/fnhum.2018.00521/full" title="https://www.frontiersin.org/articles/10.3389/fnhum.2018.00521/full">EEG Frequency Bands in Psychiatric Disorders: A Review of Resting State Studies {cite:p}`newson_eeg_2019`</a>
# 
# <a href="https://onlinelibrary.wiley.com/doi/abs/10.1111/1460-6984.12535" title="https://onlinelibrary.wiley.com/doi/abs/10.1111/1460-6984.12535">Understanding event-related potentials (ERPs) in clinical and basic language and communication disorders research: a tutorial {cite:p}`mcweeny_understanding_2020`</a>
# 

# <br/>
# 
# ### Conducting an EEG study in the context of cognitive neuroscience: a loose framework
# 
# ### A. Planning/idea to hypothesis
# 
# E.g. as a primer, we can use __<a href="http://mechanism.ucsd.edu/teaching/f18/David_Marr_Vision_A_Computational_Investigation_into_the_Human_Representation_and_Processing_of_Visual_Information.chapter1.pdf">Marr's 3 levels for the analysis of information-processing systems (p.25, {cite:p}`marr_vision_2010`)</a>__:
# 
# 
# To understand how a system operates (how a specific (beahvioral) goal is achieved, how a cognitive function can bes established etc.) we can formulate 3 distinct levels of analysis: 
#     
# **1. Computational level**: What is the goal/purpose of the system?
# 
# 
# **2. Representation/Algorithm**: how does the system solve this task? What are the representations of the input/output and how are these processed or manipulated (i.e. what is the algorithm according to which the goal is achieved)? 
# 
# 
# **3. Implementation**: the physiological implementation; i.e. what parts would a system need to solve it's purpose/to allow for the algorithm to work?
# 
# <br/>
# <br/>
# 
# _**Let's try this on a simple example:**_
# 
# E.g. we know that humans are able to differentiate pure tones based on pitch (otherwise music would be quite boring), so there has to be a system in place that enables this process, right?
#     
# -    **1.Computational level**: 
# 
#         _goal_ = differentiate between 2 soundwaves with different frequencies; 
#         
#         _purpose_=?
#    
#    
# -    **2. Representation/algorithm:**
#         
#         _input_: electric signals/neuron activation in different sensitive parts of the auditory cortex (mechanotransduction) + neural representation of time? 
#         
#         _output_:  ??? Possibly an ERP "Z"?
#         
#         _algorithm_:
#                                         
#                                         X-Y = Z (probably a bit more complex, but you get the idea)
#         
#        _(where activation of region A corresponding to frequency A = X; activation of region B corresponding to frequency B = Y)_
#        
#        
# -    **3. Implementation** (i.e. "hardware"): 
# 
#             receptor sensitive for frequency content of Soundwave (Auditory hair cells) 
#             -> mechano-electrical transduction (MET) channels 
#             -> ??? 
#             -> action potential in the connected auditory-nerve fiber 
#             -> ... 
#             -> sub-regions of the auditory cortex tuned to specific tone frequencies         
#             -> associative cortex?
#         
#  
#    _(3. {cite:p}`ahveninen_intracortical_2016`;
# <a href="https://en.wikipedia.org/wiki/Neural_encoding_of_sound">Wikipedia: Neural encoding ouf sound</a> )_
# 
# 

# ### B. Study design
# 
# Now you ideally have not only the starting question/hypothesis but a rough understanding of which areas of your research/study need to be further explored or appear to be sufficient to answer them. Next you can ask yourself how and what data/information you'll need to collect to fill in the gaps of your understanding.
# 
#    - solid understanding of theory (e.g. error monitoring; auditory processing) and biophysics?; What stimuli/paradigm to use? rate of stimuli presentation?  expected time-frame of activity? etc.
#    - what data do you expect/need? (technical aspects, such as sampling frequency, how many electrodes, additional mri scans)
#    - sample size, sample composition, how many trials/blocks/conditions?
#    - priors?; justificiaton for your alphas {cite:p}`lakens_justify_2018`?
# 

# 
# 
# ### C. Study implementation
# 
# - peregistration of methods? 
# 
# **Data collection**
#    - Standardization; testing-protocol; participant's intructions
#    - Pilot-testing
#    - Post-tesing questionnaire
#    - Experiment log
#     
# 
# **Math**
# ```{figure} https://c.tenor.com/_shmV1hUlZMAAAAd/surprised-pikachu.gif
# ---
# height: 50px
# width: 60px
# align: left
# ---
#  
# ```
# -  Preprocessing (see following chapters)
#         -> signal-to-noise ratio/power?
#    -  Analytical techniques
# 
#        - <a href="https://mne.tools/stable/auto_tutorials/time-freq/20_sensors_time_frequency.html#sphx-glr-auto-tutorials-time-freq-20-sensors-time-frequency-py">MNE: Time-frequency analysis</a>
#            
#        - <a href="https://mne.tools/stable/auto_tutorials/evoked/30_eeg_erp.html#sphx-glr-auto-tutorials-evoked-30-eeg-erp-py">MNE: EVP/ERP</a>
#        - <a href="https://mne.tools/stable/auto_tutorials/inverse/index.html">MNE: Source localization and inverses</a>
#     
#    - Followed by 
#         - classical statistics
#          - <a href="https://mne.tools/stable/auto_tutorials/stats-sensor-space/index.html">MNE: Statistical analysis of sensor data</a>
#         - machine learning
#           - <a href="https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html#sphx-glr-auto-tutorials-machine-learning-50-decoding-py">MNE: Multi-Voxel Pattern Analysis (MVPA)</a>
# 
# 
# 
# 
# **D. Making sound inferences?**
# - biologically/physiologically/theoretically informed and statistically valid?
# - does your level of certainity (presented/believed) match the degree of evidence?
# - Generalizability of sample? <a href="https://www.apa.org/monitor/2010/05/weird">(WEIRD?)</a>
# - other considerations?
# 

# <br/>
# 
# ## 4. Resources

# ### Tools for EEG-analysis
# 
# MNE
# 
# A Python based open source tool for EEG/MEG analysis, that you'll get to know in the following chapters.
# 
# <a href="https://mne.tools/stable/auto_tutorials/index.html">MNE:Tutorials</a>
# 
# 
# <a href="https://mne.tools/stable/auto_examples/index.html">MNE:Example Gallery</a>
# 
# <a href="https://mne.tools/stable/python_reference.html">MNE:API-refernces</a>
# 
# In-depth
# <a href="https://www.researchgate.net/publication/259768018_MEG_and_EEG_data_analysis_with_MNE-python">MNE: MEG and EEG data analysis with MNE-python {cite:p}`gramfort_meg_2013`</a>
# 

# ### Where to obtain EEG data ?
# 
# open-source datasets
# 
# - <a href="https://openneuro.org/search/modality/eeg">Openneuro</a>
# 
# - <a href="https://sccn.ucsd.edu/~arno/fam2data/publicly_available_EEG_data.html">USCD: list of publicly available EEG datasets</a>
# 
# - <a href="https://physionet.org/content/?topic=eeg">Physionet</a>
# 
# simulate data? (usefull for validation/testing)
# - examples:
#     <a href="https://mne.tools/dev/auto_examples/simulation/simulate_raw_data.html">MNE tutorial: Simulate raw data</a>
#      <a href="https://mne.tools/dev/auto_examples/simulation/simulate_evoked_data.htmll">MNE tutorial: Simulate evoked data</a>

# ## References
# 
# ```{bibliography}
# :filter: docname in docnames
# :style: plain
# ```
