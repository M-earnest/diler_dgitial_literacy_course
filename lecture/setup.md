# Setup: Before we get started

While most of this course will be happening online, for some parts  it will be benficial to be following along on your own machine. Below you find a list of software commonly employed in research and that will be the topic of some of our lectures.

Feel free to either follow along with this tutorial and install all of the mentioned content or if you'd prefer to take this course step-by-step you'll find that the necessary software is listed at the beginning of each of the following lectures.

Fortunately and unfortunately comitting to open source projects usually requires more in-depth knowledge than just pushing some slides onto dropbox and sending a link to your students.
To replicate the showcased content, we'll therefore be using some tools that you might not be familliar with. The following section will illustrate exatly what we need, how to install the necessary tools and where to go for further information.


**We'll need the following:**

- [**Conda**](https://conda-forge.org/) (For managing environments)
- [**Git**](https://git-scm.com) (Version Control)
- [**Visual Studio Code**](https://code.visualstudio.com/) (Text-based Editor)
- [**Jupyter**](https://jupyter.org/) (Foundation for mixed Code/ Markdown documents)
- a [**Github**](https://github.com/) account (Hosting courses)
- [**Jupyter book**](https://jupyterbook.org/en/stable/intro.html) (Foundation creating content)
- zotero



Helpful tools that are not strictly necessary but can be quite useful or make your life easier:
- [GitKraken](https://www.gitkraken.com/) (A Graphical User-Interface for the Git-version control system; Simplye downlowad and install the [Gitkraken client](https://help.gitkraken.com/gitkraken-client/how-to-install/) and [conncet it to your online Github profile](https://www.youtube.com/watch?v=5nhNfMcczlQ).)



## General things

There are a few computing requirements for the course that are absolutely necessary (beyond the few software packages you should install, described below):

- stub for listing e.g. necessary disk space, processing power, operating system etc.

If you foresee any of these being a problem please reach out to one of the instructors and enquire what steps you can take to ensure your setup is ready for the course.


You'll find the (hopefully) comprehensive set of install instructions below: The rest of this page provides more detail on installation procedures for each of the above elements, with separate instructions for each of the three major operating systems (`Windows`, `Mac OS`, and `Linux`).

### Some quick general notes on instructions

- There is no difference between `Enter` and `Return` in these instructions, so just press whatever the equivalent on your keyboard is whenever one is stated
- If you already have some of these things installed on your computer already that should (theoretically) be okay.
  However, you need to make sure that you are able to complete the steps described in [checking your install]() without issue.
  - For example, having multiple different `Python` installations on your computer can lead to incredibly frustrating issues that are very difficult to debug.
    As such, if you have already installed `Python` via some other application (not `Miniconda`/`Anaconda`), it's strongly encouraged to uninstall it before following the instructions below.

### OS-specific installation instructions

Select the tab that corresponds to your operating system and follow the instructions therein.

````{tab} Windows

**Conda**

1. Download and execute the .exe file from the [official website](https://docs.conda.io/en/latest/miniconda.html)
2. An installation window will pop up, go ahead and click through it and install into the suggested default directory.

Or follow the official [installation guide](https://conda.io/projects/conda/en/stable/user-guide/install/windows.html)

**Jupyter, Jupyter Book and nbgrader**

1. Press "Windows" and Search for "Anaconda Powershell Prompt"
2. Paste the following commands into the opened terminal:
- `conda install -c conda-forge jupyter-book`
- `conda install conda install -c conda-forge nbgrader`
- `conda install jupyter`


**Git**

Download the respective version for your system from the [official website](https://git-scm.com) and run the .exe file. Now you should be good to go.
As Git can be quite confusing for new users you may also want to additional install a GUI (graphical user interface) that makes it somewhat easier to interact and illustrate what git is actually doing/supposed to do, such as  [Gitkraken](https://www.gitkraken.com/).

**VSCode**

1. Go to https://code.visualstudio.com/ and click the download button, then run the `.exe` file.
1. Leave all the defaults during the installation with the following exception:
      - Please make sure the box labelled "Register Code as an editor for supported file types" is selected

**VSCode extensions**

1. Open VSCode
2. Press `Ctrl+Shift+P` in the new window that opens and type "Extensions: Install extensions" into the search bar that appears at the top of the screen.
   Select the appropriate entry from the dropdown menu that appears (there should be four entries; simply select the one that reads "Extensions: Install extensions").
3. A new panel should appear on the left-hand side of the screen with a search bar.
   Search for each of the following extensions and press `Install` for the first entry that appears. (The author listed for all of these extensions should be "Microsoft".)
      - Python (n.b., you will need to reload VSCode after installing this)
      - Jupyter
````

````{tab} Linux

**Conda**

1. You most likely already have a working setup, but feel free to download the appropriate file for your system from the [official website](https://docs.conda.io/en/latest/miniconda.html) and follow the [installation instrcutions](https://conda.io/projects/conda/en/stable/user-guide/install/linux.html)
.

**Jupyter, Jupyter Book and nbgrader**

1. Open a terminal
2. Paste the following commands into the opened terminal:
- `conda install -c conda-forge jupyter-book`
- `conda install conda install -c conda-forge nbgrader`
- `conda install jupyter`


**Git**

You may already have it; try opening a terminal and typing `sudo apt-get install git` (Ubuntu, Debian) or `sudo yum install git` (Fedora) inside the terminal.
If you are prompted to install it follow the instructions on-screen to do so.

**VSCode**

1. Go to https://code.visualstudio.com/ and click the download button for either the .deb (Ubuntu, Debian) or the .rpm (Fedora, CentOS) file.
1. Double-click the downloaded file to install VSCode.
   (You may be prompted to type your administrator password during the install).

**VSCode extensions**

1. Open the Visual Studio Code application.
1. Press `Ctrl+Shift+P` in the new window that opens and type "Extensions: Install extensions" into the search bar that appears at the top of the screen.
   Select the appropriate entry from the dropdown menu that appears (there should be four entries; simply select the one that reads "Extensions: Install extensions").
1. A new panel should appear on the left-hand side of the screen with a search bar.
   Search for each of the following extensions and press `Install` for the first entry that appears. (The author listed for all of these extensions should be "Microsoft".)
      - Python (n.b., you will need to reload VSCode after installing this)
      - Jupyter


````

````{tab} MacOS

**Conda**

1. Download and execute the appropriate file from the [official website](https://docs.conda.io/en/latest/miniconda.html)
2. Open a terminal at the location of the downloaded file and run: `bash Miniconda3-latest-MacOSX-x86_64.sh`

Or follow the official [installation guide](https://conda.io/projects/conda/en/stable/user-guide/install/windows.html)

**Jupyter, Jupyter Book and nbgrader**

1. Open a terminal
2. Paste the following commands into the opened terminal:
- `conda install -c conda-forge jupyter-book`
- `conda install conda install -c conda-forge nbgrader`
- `conda install jupyter`

**Git**

You may already have it!
Try opening a terminal and typing `git --version`.
If you do not see something like “git version X.XX.X” printed out, then follow these steps:

1. Follow [this link](https://sourceforge.net/projects/git-osx-installer/files/git-2.23.0-intel-universal-mavericks.dmg/download?use_mirror=autoselect) to automatically download an installer.
1. Double click the downloaded file (`git-2.23.0-intel-universal-mavericks.dmg`) and then double click the `git-2.23.0-intel-universal-mavericks.pkg` icon inside the dmg that is opened.
1. Follow the on-screen instructions to install the package.

**VSCode**

1. Go to https://code.visualstudio.com/ and click the download button.
1. Unzip the downloaded file (e.g., `VSCode-darwin-stable.zip`) and moving the resulting `Visual Studio Code` file to your Applications directory.

**VSCode extensions**

1. Open the Visual Studio Code application
1. Type `Cmd+Shift+P` and then enter "Shell command: Install 'code' command in PATH" into the search bar that appears at the top of the screen.
   Select the highlighted entry.
   A notification box should appear in the bottom-right corner indicating that the command was installed successfully.
1. Type `Cmd+Shift+P` again and then enter "Extensions: Install extensions" into the search bar.
   Select the appropriate entry from the dropdown menu that appears (there should be four entries; simply select the one that reads "Extensions: Install extensions").
1. A new panel should appear on the left-hand side of the screen with a search bar.
   Search for each of the following extensions and press `Install` for the first entry that appears. (The author listed for all of these extensions should be "Microsoft".)
      - Python (n.b., you will need to reload VSCode after installing this)
      - Jupyter



````

**Note**: If the instructions aren't working and you have spent more than 15-20 minutes troubleshooting on your own, reach out on the #help-installation channel on the Discord channel with the exact problems you're having.
One of the instructors will try and get back to you quickly to help resolve the situation.
If they're unable to help via `Discord`, you may be directed to attend one of the installation office hours.

### GitHub account

Go to https://github.com/join/ and follow the on-screen instructions to create an account.
It is a good idea to associate this with your university e-mail (if you have one) as this will entitle you to sign up for the [GitHub Student Developer Pack](https://education.github.com/pack) which comes with some nice free bonuses.



### Enter the matrix

Once you reached this point, you should be ready the enter the matrix and follow the course in your preferred way. Congrats, fantastic work!

![logo](https://media1.tenor.com/images/e5c21d98f56c4af119b4e14b6a9df893/tenor.gif?itemid=4011236)\
<sub><sup><sub><sup>https://media1.tenor.com/images/e5c21d98f56c4af119b4e14b6a9df893/tenor.gif?itemid=4011236</sup></sub></sup></sub>
S