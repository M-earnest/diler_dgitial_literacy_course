#!/usr/bin/env python
# coding: utf-8

# # The jupyter ecosystem & notebooks
# 
# 

# ### where are we now?

# 
# Roadmap/what can we expect
# 
# 

# ## Goals
# 
# * learn about jupyter notebooks, the jupyter eco-system and their role in academia
# * learn basic and efficient usage of the `jupyter ecosystem` & `notebooks`
#     * what is `Jupyter` & how to utilize `jupyter notebooks`

# ## Before we get started...
# 
# 
#     
# We're going to be exploring the `Jupyter notebooks` interface in this section of the course!
# 
# If you've followed the [Setup](https://m-earnest.github.io/diler_dgitial_literacy_course/setup.html) and installed conda and jupyter, you can simply open a notebook yoursel by either:
# 
# A. Opening the Anaconda application and selecting the Jupyter Notebooks tile
# 
# B. Or opening a terminal/shell type `jupyter notebook` and hit `enter`. If you're not automatically directed to a webpage copy the URL (`https://....`) printed in the `terminal` and paste it in your `browser`
# 
# 
# 
# ### Note on interactive Mode
# 
# As this website is build on Jupyter Notebooks you can click also on the small rocket at the top of this website, select `Live code` (and wait a bit) and this site will become interactive.
#     
# ![Launch MyBinder](https://raw.githubusercontent.com/felixkoerber/jb/main/static/Launch_binder.png)
# 
# 
# Following you can try to run the `code cells` below, by clicking on the "run" button, that appears beneath them.
# 
# Some functionality of this notebooks can't be demonstrated using the live code implementation you can therefore either [download the course content]() or open this notebook via Binder, i.e. by clicking on the rocket and select `Binder`. This will open an online Jupyter-lab session where you can find this notebook by follow the folder strcuture that will be opened on the right hand side via `lecture` -> `content` and click on `intro_jupyter.ipynb`.
# 

# ## To Jupyter & beyond
# 
# <img align="center" src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/jupyter_example.png" alt="logo" title="jupyter" width="900" height="400" /> 

# 
# <img align="center" src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/jupyter_ecosystem.png" alt="logo" title="jupyter" width="500" height="200" /> 
# 
# 
# </br>
# </br>
# </br>
# 
# **Jupyter notebooks** are a widely-used, open-source tool for interactive computing. In the following parts we'll do a deep dive on how to use jupyter notebooks, but let's first discuss why and for what you'd use jupyter in your research.
# 
# The main benefits of Jupyter notebooks lies in the ability to combine live code, equations, visualizations, rich media presentations, and narrative text in a single document. This makes it easy to create engaging, interactive content to communicate complex concepts, such as a research workflow. Notebooks are further great 
# 
#     - data visualization and exploration, 
#     - documentation
#     - teaching/learning 
#     - presentation of results
# 
# Due to this **Jupyter Notebookss** have become the community standard for communicating and performing interactive computing.
# 
# **Noteebooks** can be freely shared, using a few other tools that you'll get to know they can even be hosted online in the form of a website, i.e. as is this course.
# - great for learning, teaching, documentation, "living publications"
# 
# 
# 
# **And What is Jupyter?**
# 
# - a community of people
#  
# - an ecosystem of open tools and standards for interactive computing
# 
# - language-agnostic and modular
#  
# - empower people to use other open tools
# 
# Jupyter stands for `Julia, Python, and R.`These were the initial programming language that the Jupyter framework was developed for. Since then a number of other languages have been incorporated such as Fortran, Stata and Matlab, although these require the installation of additional software before being used. For more information on this you can follow the [official documentation on Jupyte Kernels](https://docs.jupyter.org/en/latest/projects/kernels.html).
# 
# 

# ## The interface
# 
# ### The Files tab
# 
# When you first open the notebook application you will be brought to the `files tab`.
# 
# The `files tab` provides an interactive view of the portion of the `filesystem` which is accessible by the `user`. This is typically rooted by the directory in which the notebook server was started.
# 
# The top of the `files list` displays the structure of the `current directory`. It is possible to navigate the `filesystem` by clicking on these `breadcrumbs` or on the `directories` displayed in the `notebook list`.
# 
# <img align="center" src="../static/jupyter_tabs.png" alt="picture of jupyter files tab" title="files tab" width="900" height="300" />
# 
# ### Creating a notebook
# 
# A new `notebook` can be created by clicking on the `New dropdown button` at the top of the list, and selecting the desired [`language kernel`](https://docs.jupyter.org/en/latest/projects/kernels.html). We'll be using Python, but Kernels for a plethora of other languages exist. An comprehenisve list of Jupyter Kernels can be found [here](https://github.com/jupyter/jupyter/wiki/Jupyter-kernels).
# 
# `Notebooks` can also be `uploaded` to the `current directory` by dragging a `notebook` file onto the list or by clicking the `Upload button` at the top of the list.
# 
# 

# ### The Notebook
# 
# When a `notebook` is opened, a new `browser tab` will be created which presents the `notebook user interface (UI)`. This `UI` allows for `interactively editing` and `running` the `notebook document`.
# 
# A new `notebook` can be created from the `dashboard` by clicking on the `Files tab`, followed by the `New dropdown button`, and then selecting the `language` of choice for the `notebook`.
# 
# An `interactive tour` of the `notebook UI` can be started by selecting `Help` -> `User Interface Tour` from the `notebook menu bar`.

# ### Header
# 
# At the top of the `notebook document` is a `header` which contains the `notebook title`, a `menubar`, and `toolbar`. This `header` remains `fixed` at the top of the screen, even as the `body` of the `notebook` is `scrolled`. The `title` can be edited `in-place` (which renames the `notebook file`), and the `menubar` and `toolbar` contain a variety of actions which control `notebook navigation` and `document structure`.
# 
# <img align="center" src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/notebook_header_4_0.png" alt="logo" title="jupyter" width="600" height="100" /> 

# ### Body
# 
# The `body` of a `notebook` is composed of `cells`. Each `cell` contains either `markdown`, `code input`, `code output`, or `raw text`. `Cells` can be included in any order and edited at-will, allowing for a large amount of flexibility for constructing a narrative.
# 
# - `Markdown cells` - These are used to build a `nicely formatted narrative` around the `code` in the document. The majority of this lesson is composed of `markdown cells`.
# - to get a `markdown cell` you can either select the `cell` and use `esc` + `m` or via `Cell -> cell type -> markdown`
# 
# <img align="center" src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/notebook_body_4_0.png" alt="logo" title="jupyter" width="700" height="200" />

# - `Code cells` - These are used to define the `computational code` in the `document`. They come in `two forms`: 
#     - the `input cell` where the `user` types the `code` to be `executed`,  
#     - and the `output cell` which is the `representation` of the `executed code`. Depending on the `code`, this `representation` may be a `simple scalar value`, or something more complex like a `plot` or an `interactive widget`.
# - to get a `code cell` you can either select the `cell` and use `esc` + `y` or via `Cell -> cell type -> code`
# 
#     
# <img align="center" src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/notebook_body_4_0.png" alt="logo" title="jupyter" width="700" height="200" />
#     

# - `Raw cells` - These are used when `text` needs to be included in `raw form`, without `execution` or `transformation`.
# 
# <img align="center" src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/notebook_body_4_0.png" alt="logo" title="jupyter" width="700" height="200" />
#  

# 
# ### Note on interactive Mode
# 
# If you've activated the interactive mode of this website you can try to run the `code cell` below, by clicking into it and pressing `ctrl + enter`.
# 

# In[2]:


print('hello')


# ### Modality
# 
# The `notebook user interface` is `modal`. This means that the `keyboard` behaves `differently` depending upon the `current mode` of the `notebook`. A `notebook` has `two modes`: `edit` and `command`.
# 
# `Edit mode` is indicated by a `green cell border` and a `prompt` showing in the `editor area`. When a `cell` is in `edit mode`, you can type into the `cell`, like a `normal text editor`.
# 
# <img align="center" src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/edit_mode.png" alt="logo" title="jupyter" width="700" height="100" /> 

# `Command mode` is indicated by a `grey cell border`. When in `command mode`, the structure of the `notebook` can be modified as a whole, but the `text` in `individual cells` cannot be changed. Most importantly, the `keyboard` is `mapped` to a set of `shortcuts` for efficiently performing `notebook and cell actions`. For example, pressing `c` when in `command` mode, will `copy` the `current cell`; no modifier is needed.
# 
# <img align="center" src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/command_mode.png" alt="logo" title="jupyter" width="700" height="100" /> 

# ### Mouse navigation
# 
# The `first concept` to understand in `mouse-based navigation` is that `cells` can be `selected by clicking on them`. The `currently selected cell` is indicated with a `grey` or `green border depending` on whether the `notebook` is in `edit or command mode`. Clicking inside a `cell`'s `editor area` will enter `edit mode`. Clicking on the `prompt` or the `output area` of a `cell` will enter `command mode`.
# 
# The `second concept` to understand in `mouse-based navigation` is that `cell actions` usually apply to the `currently selected cell`. For example, to `run` the `code in a cell`, select it and then click the  `Run button` in the `toolbar` or the `Cell` -> `Run` menu item. Similarly, to `copy` a `cell`, select it and then click the `copy selected cells  button` in the `toolbar` or the `Edit` -> `Copy` menu item. With this simple pattern, it should be possible to perform nearly every `action` with the `mouse`.
# 
# `Markdown cells` have one other `state` which can be `modified` with the `mouse`. These `cells` can either be `rendered` or `unrendered`. When they are `rendered`, a nice `formatted representation` of the `cell`'s `contents` will be presented. When they are `unrendered`, the `raw text source` of the `cell` will be presented. To `render` the `selected cell` with the `mouse`, click the  `button` in the `toolbar` or the `Cell` -> `Run` menu item. To `unrender` the `selected cell`, `double click` on the `cell`.

# ### Keyboard Navigation
# 
# The `modal user interface` of the `IPython Notebook` has been optimized for efficient `keyboard` usage. This is made possible by having `two different sets` of `keyboard shortcuts`: one set that is `active in edit mode` and another in `command mode`.
# 
# The most important `keyboard shortcuts` are `Enter`, which enters `edit mode`, and `Esc`, which enters `command mode`.
# 
# In `edit mode`, most of the `keyboard` is dedicated to `typing` into the `cell's editor`. Thus, in `edit mode` there are relatively `few shortcuts`. In `command mode`, the entire `keyboard` is available for `shortcuts`, so there are many more possibilities.
# 
# The following images give an overview of the available `keyboard shortcuts`. These can viewed in the `notebook` at any time via the `Help` -> `Keyboard Shortcuts` menu item.

# <img align="center" src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/notebook_shortcuts_4_0.png" alt="logo" title="jupyter" width="500" height="500" /> 

# ### The following shortcuts have been found to be the most useful in day-to-day tasks:
# 
# - Basic navigation: `enter`, `shift-enter`, `up/k`, `down/j`
# - Saving the `notebook`: `s`
# - `Cell types`: `y`, `m`, `1-6`, `r`
# - `Cell creation`: `a`, `b`
# - `Cell editing`: `x`, `c`, `v`, `d`, `z`, `ctrl+shift+-`
# - `Kernel operations`: `i`, `.`
# 
# 
# Additionally, you should get in the habit of using `Tab` to auto-complete your code. This not only speeds up things, but also makes sure that your variable and file names or the specific function you want to use is actually spelled correctly. In `edit mode` you should further get used to use the `Ctrl + down/up/left/right` shortcuts to quickly navigate trough your code cells.

# ## Adding formated Text to a Notebook
# 
# ### Markdown Cells
# 
# `Text` can be added to `IPython Notebooks` using `Markdown cells`. 
# 
# `Markdown` is a popular `markup language` that is a `superset of HTML`. Using markdown allows us to simply structure texts in Jupyter notebooks.
# 
# Its specification can be found here: [Markdown](https://daringfireball.net/projects/markdown/basics)
# 
# Following we will go over some bits of basic Markdown formatting. For a more detailed view, check out markdownguide.com as one of many ressources.
# 
# _**Note**: Markdown cells become "rendered", when you run them like we did with the code cell above. You can view the `source` of a `cell` by `double clicking` on it, or while the `cell` is selected in `command mode`, press `Enter` to edit it. Once a `cell` has been `edited`, use `Shift-Enter` to `re-render` it. Unfortunately this functionaliyt has as of yet not been extended to the live code implementation and can't therefore be demonstated here, so you'll have to try this out in a notebook for yourself or open this notebook via the binder application as explained above._

# ### Formatting Text
# 
# Let's start with the fundamentals:
# 
# #### Italic
# To make a text _italic_ add `_` or `*` before and after the word: 
# |Syntax   | Output|
# |---|---|
# |`_italic_`| _italic_ |
# |`*italic*` | *italic*|
# 
# To make a text *bold* add "*" before and after the word: `*bold*`= *bold*
# 
# #### Bold
# To make a text **bold** add `__` or `**` before and after the word: 
# 
# |Syntax   | Output|
# |---|---|
# |`__bold__`| __bold__ |
# |`**bold**` | **bold**|
# 
# 
# ___
# ### (Nested) Lists
# You can build nested itemized or enumerated lists using either `*` or `-` before a word
# 
# * One
#     - Sublist
#         - This
#   - Sublist
#         - That
#         - The other thing
# * Two
#   - Sublist
# * Three
#   - Sublist
# 
# You also create numbered lists by using `1.` etc. before your point:
# 
# 1. Here we go
#     1. Sublist
#     2. Sublist
# 2. There we go
# 3. Now this
#   
# ### Horizontal Lines
# You can add horizontal rules using three underscores `___` resulting in:
# 
# ---

# ### Blockquotes
# To create a blockquote, it is as simple as putting a `>` before a text.
# 
# Here is a blockquote(i.e the [Zen of Python](https://en.wikipedia.org/wiki/Zen_of_Python)) :
# 
# > Beautiful is better than ugly.
# > Explicit is better than implicit.
# > Simple is better than complex.
# > Complex is better than complicated.
# > Flat is better than nested.
# > Sparse is better than dense.
# > Readability counts.
# > Special cases aren't special enough to break the rules.
# > Although practicality beats purity.
# > Errors should never pass silently.
# > Unless explicitly silenced.
# > In the face of ambiguity, refuse the temptation to guess.
# > There should be one-- and preferably only one --obvious way to do it.
# > Although that way may not be obvious at first unless you're Dutch.
# > Now is better than never.
# > Although never is often better than *right* now.
# > If the implementation is hard to explain, it's a bad idea.
# > If the implementation is easy to explain, it may be a good idea.
# > Namespaces are one honking great idea -- let's do more of those!

# ### Headings
# 
# You can add headings using Markdown's syntax by adding `#` before your heading. You can vary the heading level by increasing the amount of Hash signs:
# 
# <pre>
# # Heading 1
# 
# # Heading 2
# 
# ## Heading 2.1
# 
# ## Heading 2.2
# </pre>

# ### Embedded code
# 
# You can embed code meant for illustration instead of execution in Python by adding backticks  **\`**  
# 
# before and after statements in a markdown cell:
# 
# So this markdown text:
# ```
# `def f(x):` `return x**2`
# ```
# 
# becomes this rendered text:
# 
# `def f(x):` `return x**2`
# 
# 
# Since you need to add this line by line, this might ruin your code formatting. Instead, consider using the HTML formatting, adding `<code>` before and `</code>` after your code:
# 
# so that:
# 
# ```
# <code>def f(x): return x**2</code>
# ```
# results in:
# 
# <code>def f(x): return x**2</code>

# ### General HTML
# 
# Again because `Markdown` is a `superset of HTML` you can even add things like `HTML tables`.
# 
# So this `html code` will generate the `table` below.
# 
# ```
# <table>
# <tr>
# <th>Header 1</th>
# <th>Header 2</th>
# </tr>
# <tr>
# <td>row 1, cell 1</td>
# <td>row 1, cell 2</td>
# </tr>
# <tr>
# <td>row 2, cell 1</td>
# <td>row 2, cell 2</td>
# </tr>
# </table>
# 
# ```
# <table>
# <tr>
# <th>Header 1</th>
# <th>Header 2</th>
# </tr>
# <tr>
# <td>row 1, cell 1</td>
# <td>row 1, cell 2</td>
# </tr>
# <tr>
# <td>row 2, cell 1</td>
# <td>row 2, cell 2</td>
# </tr>
# </table>

# ## Adding Media to your notebook

# ### Images
# Using Jupyter Notebooks, there few different ways to include images.
# 
# The easiest way is to add images to your project is by referencing files either online via link or on your file system by using the pathname for the file you want to reference (e.g. `/Users/username/Desktop/image.jpg` or `C:\Users\Username\Desktop`). 
# 
# To add images simply use on of the following line of Markdown in a markdown cell:
# 
# `![alternative_imagetext](https://YOURGITHUBNAME.github.io/PROJECTNAME/_static/FILENAME.png)`
# 
# Resulting, for example, in:
# 
# ![alternative_imagetext](https://felixkoerber.github.io/jb/_static/logo.png)
# 
# 
# As you are able to use essentially any imagelink, you can add images that are hosted through other websites, too!
# 
# 
# If you want more control about the size, orientation and more, you will have to rely on HTML tags. E.g. the following line of code will present the same image as above only downscaled to a width of 200 pixels.
# 
# 
# `<img src="(https://YOURGITHUBNAME.github.io/PROJECTNAME/_static/FILENAME.png" alt="alternative_imagetext" class="bg-primary mb-1" width="200px">`
# 
# 
# which results in:
# 
# <img src="https://felixkoerber.github.io/jb/_static/logo.png" alt="logo" class="bg-primary mb-1" width="200px">
# 
# 
# 
# </br>
# </br>
# </br>
# </br>
# 
# **The code above is made up of multiple html tags, that each control a separate parameter of the image you want to control:**
# 
# `<img>` is an HTML tag used to display an image on a web page.
# 
# `src="(https://YOURGITHUBNAME.github.io/PROJECTNAME/_static/FILENAME.png"` is an attribute that specifies the URL or path to the image file you want to display. In this case, it points to an image file on GitHub with a specific URL that you will need to replace with your own GitHub account name, project name, and file name.
# 
# `alt="alternative_imagetext"` is an attribute that provides alternative text for the image. This text is used by screen readers for visually impaired users, and it is also displayed when the image cannot be loaded for some reason.
# 
# `class="bg-primary mb-1"` is an attribute that defines the CSS class or classes to apply to the image. In this case, the class bg-primary sets the background color of the image to a primary color, and mb-1 adds a margin-bottom of 1 unit to the image.
# 
# `width="200px"` is an attribute that sets the width of the image to 200 pixels. This can be adjusted to suit your needs.
# 
# Simply copy-paste the above line of code and adjust it to your need as necessary to emebdd every kind of image into your notebooks.
# 
# **For a more in-depth exploration of the `img` tag checkout the [w3schools html tutorial](https://www.w3schools.com/html/html_images.asp)**
# 

# ### Videos, Presentations, GIFs, and more: iframes
# 
# By utilizing `html`, wen can integrate almost every type of media, including - but not limited to - `Videos, Presentations, GIFs, Maps` and `Images` via the `iframes` keyword.
# 
# ```{note}
# `iframes` or *inlineframes* are HTML-elements, that can be used to embed content from different websites.
# ```
# 
# Inlineframes are structured are implemented like this:
# 
# `<iframe src="http://www.example.com/" height="100" width="200" name="iframename">Alternative title</iframe>`
# 
# 
# where:
# 
# `<iframe>` is an HTML tag used to embed another webpage within the current webpage.
# 
# `src="http://www.example.com/"` is an attribute that specifies the URL of the webpage you want to embed. In this case, it points to a webpage with the URL "http://www.example.com/". You can replace this URL with the URL of the webpage you want to embed.
# 
# `height="100"` is an attribute that sets the height of the iframe element in pixels. In this case, it is set to a height of 100 pixels. You can adjust this value to change the height of the iframe element.
# 
# `width="200"` is an attribute that sets the width of the iframe element in pixels. In this case, it is set to a width of 200 pixels. You can adjust this value to change the width of the iframe element.
# 
# `name="iframename"` is an attribute that sets the name of the iframe element. This can be useful if you want to target the iframe element with JavaScript or CSS.
# 
# `Alternative title` is the content that will be displayed in the iframe element if the browser does not support iframes. It can also be used by screen readers for visually impaired users.
# 
# Thus, the `iframe` tag essentially follows the same structuring like the html-image integration.
# 
# **To quickly generate this code for the media of your choice, you can use tool like the [iframe-generator](https://www.iframe-generator.com/)**, where you add the media url, can change various settings and can then generate the iframe code.
# 
# 
# Let's go over different types of media embeddings:
# 
# ___
# 
# 

# 
# ### GIFs
# 
# To implement GIFs in Jupyter Books, you can also use HTML to embed a GIF in Jupyter Books. Here's an example:
# 
# `<iframe src="link/to/your/gif/file.gif" width="240" height="200" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>'`
# 
# Again, replace "link/to/your/gif/file.gif" with the URL or file path to your GIF.
# 
# This could result in:
# 
# <iframe src="https://giphy.com/embed/l5s71uAp3CzKwxwkoZ" width="240" height="200" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>
# 
# ```{note}
# Note that if you're using a local file path, the path should be relative to the location of the notebook or Markdown file that you're embedding the GIF in. If you're using a URL, it should be a direct link to the GIF file.
# ```
# 
# ___

# ### Working with local files 
# 
# 
# If you have `local files`, such as a picture you'd like to include in your `Notebook directory`, you can refer to these `files` in `Markdown cells` directly:
# 
#     [subdirectory/]<filename>
# 
# For example, in the `static folder` of this course, we have the `logo`:
# 
#     <img src="static/pfp_logo.png" />
# 
# <img src="../static/pfp_logo.png" width=300 />
# 
# 
# These do not `embed` the data into the `notebook file`, and require that the `files` exist when you are viewing the `notebook`.

# ### Security of local files
# 
# Note that this means that the `IPython notebook server` also acts as a `generic file server` for `files` inside the same `tree` as your `notebooks`. Access is not granted outside the `notebook` folder so you have strict control over what `files` are `visible`, but for this reason **it is highly recommended that you do not run the notebook server with a notebook directory at a high level in your filesystem (e.g. your home directory)**.
# 
# When you run the `notebook` in a `password-protected` manner, `local file` access is `restricted` to `authenticated users` unless `read-only views` are active.

# ### Markdown attachments
# 
# Since `Jupyter notebook version 5.0`, in addition to `referencing external files` you can `attach a file` to a `markdown cell`. To do so `drag` the `file` from e.g. the `browser` or local `storage` in a `markdown cell` while `editing` it:
# 
# `![pfp_logo.png](attachment:pfp_logo.png)`
# 
# ![pfp_logo.png](attachment:pfp_logo.png)

# `Files` are stored in `cell metadata` and will be `automatically scrubbed` at `save-time` if not `referenced`. You can recognize `attached images` from other `files` by their `url` that starts with `attachment`. For the `image` above:
# 
#     ![pfp_logo.png](attachment:pfp_logo.png)
# 
# Keep in mind that `attached files` will `increase the size` of your `notebook`.
# 
# You can manually edit the `attachement` by using the `View` > `Cell Toolbar` > `Attachment` menu, but you should not need to.

# ### Code cells
# 
# When executing code in `IPython`, all valid `Python syntax` works as-is, but `IPython` provides a number of `features` designed to make the `interactive experience` more `fluid` and `efficient`. First, we need to explain how to run `cells`. Try to run the `cell` below!

# In[1]:


import pandas as pd

print("Hi! This is a cell. Click on it and press the ▶ button above to run it")


# You can also run a cell with `Ctrl+Enter` or `Shift+Enter`. Experiment a bit with that.

# ## Practical excercises:
# 
# If you've opened this page via Binder as described above or downloaded the course materials follow along with the next couple of excerices. You can also open a new notebook as described above and copy the following code cells into your notebook.

# ### Tab Completion
# 
# 
# 
# One of the most useful things about `Jupyter Notebook` is its tab completion.
# 
# Click just after `read_csv`( in the cell below and press `Shift+Tab` 4 times, slowly. Note that if you're using `JupyterLab` you don't have an additional help box option.

# In[ ]:


pd.read_csv(


# After the first time, you should see this:
# 
# <img align="center" src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/jupyter_tab-once.png" alt="logo" title="jupyter" width="700" height="200" /> 

# After the second time:
# 
# <img align="center" src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/jupyter_tab-twice.png" alt="logo" title="jupyter" width="500" height="200" /> 

# After the fourth time, a big help box should pop up at the bottom of the screen, with the full documentation for the `read_csv` function:
# 
# <img align="center" src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/jupyter_tab-4-times.png" alt="logo" title="jupyter" width="700" height="300" /> 
# 
# This is amazingly useful. You can think of this as "the more confused I am, the more times I should press `Shift+Tab`".

# Okay, let's try `tab completion` for `function names`!

# In[ ]:


pd.r


# You should see this:
# 
# <img align="center" src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/jupyter_function-completion.png" alt="logo" title="jupyter" width="300" height="200" /> 

# ## Get Help
# 
# There's an additional way on how you can reach the help box shown above after the fourth `Shift+Tab` press. Instead, you can also use `obj`? or `obj`?? to get help or more help for an object.

# In[2]:


get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# ## Writing code
# 
# Writing code in a `notebook` is pretty normal.

# In[3]:


def print_10_nums():
    for i in range(10):
        print(i)


# In[4]:


print_10_nums()


# If you messed something up and want to revert to an older version of a code in a cell, use `Ctrl+Z` or to go than back `Ctrl+Y`.
# 
# For a full list of all keyboard shortcuts, click on the small `keyboard icon` in the `notebook header` or click on `Help` > `Keyboard Shortcuts`.

# ### The interactive workflow: input, output, history
# 
# `Notebooks` provide various options for `inputs` and `outputs`, while also allowing to access the `history` of `run commands`.

# In[5]:


2+10


# you can access the output of the most recent cell by simply using `_`. The underscore variable will continously be updated everytime you run a cell

# In[6]:


_+10


# In[7]:


_


# same goes for the second-to-last output via a double underscore, the third-to-last output via a triple underscore and so on

# In[8]:


___


# You can suppress the `storage` and `rendering` of `output` if you append `;` to the last `cell` (this comes in handy when plotting with `matplotlib`, for example):

# In[9]:


10+20;


# In[10]:


_


# As this notation get's messy real quick, use other ways to access earlier outputs using the `_N` and `Out[N]` variables:

# In[11]:


Out[10]


# In[12]:


_10 == Out[10]


# Previous inputs are available, too:

# In[13]:


In[11]


# In[14]:


_i


# and to be even more explicit use `%history`

# In[15]:


get_ipython().run_line_magic('history', '-n 1-5')


# ### Accessing the underlying operating system
# 
# Through `notebooks` you can also access the underlying `operating system` and `communicate` with it as you would do in e.g. a `terminal` via the `bash` programming language:

# In[16]:


get_ipython().system('pwd')


# In[17]:


files = get_ipython().getoutput('ls')
print("My current directory's files:")
print(files)


# In[18]:


get_ipython().system('echo $files')


# ### Magic functions
# 
# `IPython` has all kinds of `magic functions`. `Magic functions` are prefixed by `%` or `%%,` and typically take their `arguments` without `parentheses`, `quotes` or even `commas` for convenience. `Line magics` take a single `%` and `cell magics` are prefixed with two `%%`.

# Some useful magic functions are:
# 
# Magic Name | Effect
# ---------- | -------------------------------------------------------------
# %env       | Get, set, or list environment variables
# %pdb       | Control the automatic calling of the pdb interactive debugger
# %pylab     | Load numpy and matplotlib to work interactively
# %%debug    | Activates debugging mode in cell
# %%html     | Render the cell as a block of HTML
# %%latex    | Render the cell as a block of latex
# %%sh       | %%sh script magic
# %%time     | Time execution of a Python statement or expression
# 
# You can run `%magic` to get a list of `magic functions` or `%quickref` for a reference sheet.

# In[19]:


get_ipython().run_line_magic('magic', '')


# `Line` vs `cell magics`:

# In[20]:


get_ipython().run_line_magic('timeit', 'list(range(1000))')


# In[21]:


get_ipython().run_cell_magic('timeit', '', 'list(range(10))\nlist(range(100))\n')


# `Line magics` can be used even inside `code blocks`:

# In[22]:


for i in range(1, 5):
    size = i*100
    print('size:', size, end=' ')
    get_ipython().run_line_magic('timeit', 'list(range(size))')


# `Magics` can do anything they want with their input, so it doesn't have to be valid `Python`:

# In[23]:


get_ipython().run_cell_magic('bash', '', 'echo "My shell is:" $SHELL\necho "My disk usage is:"\ndf -h\n')


# Another interesting `cell magic`: create any `file` you want `locally` from the `notebook`:

# In[25]:


get_ipython().run_cell_magic('writefile', 'test.txt', 'This is a test file!\nIt can contain anything I want...\n\nAnd more...\n')


# In[26]:


get_ipython().system('cat test.txt')


# Let's see what other `magics` are currently defined in the `system`:

# In[27]:


get_ipython().run_line_magic('lsmagic', '')


# ## Writing latex 
# 
# Let's use `%%latex` to render a block of `latex`:

# In[28]:


get_ipython().run_cell_magic('latex', '', '$$F(k) = \\int_{-\\infty}^{\\infty} f(x) e^{2\\pi i k} \\mathrm{d} x$$\n')


# ### Running normal Python code: execution and errors
# 
# Not only can you input normal `Python code`, you can even paste straight from a `Python` or `IPython shell session`:

# In[29]:


# Fibonacci series:
# the sum of two elements defines the next
a, b = 0, 1
while b < 10:
    print(b)
    a, b = b, a+b


# In[30]:


for i in range(10):
    print(i, end=' ')
    


# And when your code produces errors, you can control how they are displayed with the `%xmode` magic:

# In[31]:


get_ipython().run_cell_magic('writefile', 'mod.py', '\ndef f(x):\n    return 1.0/(x-1)\n\ndef g(y):\n    return f(y+1)\n')


# Now let's call the function `g` with an argument that would produce an error:

# In[32]:


import mod
mod.g(0)


# In[33]:


get_ipython().run_line_magic('xmode', 'plain')
mod.g(0)


# In[34]:


get_ipython().run_line_magic('xmode', 'verbose')
mod.g(0)


# The default `%xmode` is "context", which shows additional context but not all local variables.  Let's restore that one for the rest of our session.

# In[35]:


get_ipython().run_line_magic('xmode', 'context')


# ## Running code in other languages with special `%%` magics

# In[36]:


get_ipython().run_cell_magic('perl', '', '@months = ("July", "August", "September");\nprint $months[0];\n')


# In[37]:


get_ipython().run_cell_magic('ruby', '', 'name = "world"\nputs "Hello #{name.capitalize}!"\n')


# ### Raw Input in the notebook
# 
# Since `1.0` the `IPython notebook web application` supports `raw_input` which for example allow us to invoke the `%debug` `magic` in the `notebook`:

# In[38]:


mod.g(0)


# In[39]:


get_ipython().run_line_magic('debug', '')


# Don't forget to exit your `debugging session` using the `exit()` function. `Raw input` can of course be used to ask for `user input`:

# In[41]:


enjoy = input('Are you enjoying this tutorial? ')
print('Answer:', enjoy)


# ### Plotting in the notebook
# 
# `Notebooks` support a variety of fantastic `plotting options`, including `static` and `interactive` graphics. This `magic` configures `matplotlib` to `render` its `figures` `inline`:

# In[42]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[43]:


import numpy as np
import matplotlib.pyplot as plt


# In[44]:


x = np.linspace(0, 2*np.pi, 300)
y = np.sin(x**2)
plt.plot(x, y)
plt.title("A little chirp")
fig = plt.gcf()  # let's keep the figure object around for later...


# In[45]:


import plotly.figure_factory as ff

# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2
x4 = np.random.randn(200) + 4

# Group data together
hist_data = [x1, x2, x3, x4]

group_labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
fig.show()


# ## The IPython kernel/client model

# In[46]:


get_ipython().run_line_magic('connect_info', '')


# We can connect automatically a [Qt Console](https://qtconsole.readthedocs.io/en/stable/index.html) to the currently running kernel with the `%qtconsole` magic, or by typing `ipython console --existing <kernel-UUID>` in any terminal:

# In[47]:


get_ipython().run_line_magic('qtconsole', '')


# ## Saving a Notebook
# 
# `Jupyter Notebooks` `autosave`, so you don't have to worry about losing code too much. At the top of the page you can usually see the current save status:
# 
# `Last Checkpoint: 2 minutes ago (unsaved changes)`
# `Last Checkpoint: a few seconds ago (autosaved)`
# 
# If you want to save a notebook on purpose, either click on `File` > `Save` and `Checkpoint` or press `Ctrl+S`.

# ## To Jupyter & beyond
# 
# <img align="center" src="https://raw.githubusercontent.com/PeerHerholz/ML-DL_workshop_SynAGE/master/lecture/static/jupyter_example.png" alt="logo" title="jupyter" width="800" height="400" /> 

# 1. Open a terminal

# 2. Type `jupyter lab`

# 3. If you're not automatically directed to a webpage copy the URL printed in the terminal and paste it in your browser

# 4. Click "New" in the top-right corner and select "Python 3"

# 5. You have a `Jupyter notebook` within `Jupyter lab`!

# ## Excercise
# 
# - Generate a new `jupyter notebook` with
#     -   `3 different cells`:
#             - 1 rendered markdown cell within which you name your favorite muscial artists and describe why you like them via max. 2 sentences
#             - 1 code cell with an equation (e.g. `1+1`, `(a+b)/(c+d)`, etc.)
#             - 1 raw cell with your favorite snack 
#     - **optional**: try to include a picture of your favorite animal via the methods learned above
# 
# 

#     
# 
# ### Additional materials
# 
# 
# 
# 
# 

# ##  Achknowledgments
# 
# 
# <br>
# 
# - most of what you’ll see within this lecture was prepared by Ross Markello, Michael Notter and Peer Herholz and further adapted for this course by Peer Herholz, Michael Ernst & Felix Körber
# - based on Tal Yarkoni's ["Introduction to Python" lecture at Neurohackademy 2019](https://neurohackademy.org/course/introduction-to-python-2/)
# - based on [IPython notebooks from J. R. Johansson](http://github.com/jrjohansson/scientific-python-lectures)
# - based on http://www.stavros.io/tutorials/python/ & http://www.swaroopch.com/notes/python
# - based on https://github.com/oesteban/biss2016 &  https://github.com/jvns/pandas-cookbook
# 
# 
# [Michael Ernst](https://github.com/M-earnest)  
# Phd student - [Fiebach Lab](http://www.fiebachlab.org/), [Neurocognitive Psychology](https://www.psychologie.uni-frankfurt.de/49868684/Abteilungen) at [Goethe-University Frankfurt](https://www.goethe-university-frankfurt.de/en?locale=en)
# 
# 
# [Peer Herholz (he/him)](https://peerherholz.github.io/)  
# Research affiliate - [NeuroDataScience lab](https://neurodatascience.github.io/) at [MNI](https://www.mcgill.ca/neuro/)/[MIT](https://www.mit.edu/)  
# Member - [BIDS](https://bids-specification.readthedocs.io/en/stable/), [ReproNim](https://www.repronim.org/), [Brainhack](https://brainhack.org/), [Neuromod](https://www.cneuromod.ca/), [OHBM SEA-SIG](https://ohbm-environment.org/), [UNIQUE](https://sites.google.com/view/unique-neuro-ai)  
# 
# <img align="left" src="https://raw.githubusercontent.com/G0RELLA/gorella_mwn/master/lecture/static/Twitter%20social%20icons%20-%20circle%20-%20blue.png" alt="logo" title="Twitter" width="32" height="20" /> <img align="left" src="https://raw.githubusercontent.com/G0RELLA/gorella_mwn/master/lecture/static/GitHub-Mark-120px-plus.png" alt="logo" title="Github" width="30" height="20" />   &nbsp;&nbsp;@peerherholz 
