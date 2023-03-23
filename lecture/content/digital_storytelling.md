# Digital Storytelling

````{margin}
```{warning}
These pages are currently under construction and will be updated continuously.
Please visit these pages again in the next few weeks for further information.
````


    
Digital storytelling as a broader term involves the use of `technology to convey stories`. There are various mediums to tell these stories, such as `text` on a website or social media platform, `images` and `narration` in a video, or `audio narration` in a podcast.

Digital stories are not simply presentations of facts with accompanying visuals; they are carefully crafted narratives that take the audience on a journey. Like novels or documentaries, digital stories have elements such as `plot, characters, and themes`.

#### Digital Storytelling in Education

With an everincreasing use of digital media, digital storytelling is of high importance in the field of education, to facilitate engagement of scholars by creating dynamic and interactive narratives. The use of multimedia elements like images, videos and narration can lead to a more immersive experience, promoting creativity and critical thinking.
Furthermore, digital storytelling helps to convey essential skills of digital literacy!

#### Living Documents

Expaning on digital storytelling, there is a growing usage of "living documents" which utilize technology to create `dynamic, constantly evolving narratives`. These living documents are documents that can be `updated in real-time`, with new information and perspectives added as they emerge.

In the realm of science, living documents are becoming increasingly important tools for sharing information and promoting collaboration among researchers. For example, scientists may use living documents to create a `comprehensive database of research studies` on a particular topic, which can be updated in real-time as new studies are published. This can help researchers stay up-to-date on the latest findings and identify areas where further research is needed.

Living documents are also particular relevant in the field of open science, where researchers `share their data and methodologies openly and collaboratively`. Living documents can be used to create a shared repository of data, which can be updated as new data becomes available. This can help promote transparency and collaboration, as researchers can work together to analyze and interpret the data in new and innovative ways.

There are many potential avenues to create living documents.


#### Tips on Creating engaging Media 

There are a couple things you can look out for using better digital storytelling:


>For further information on how to use digital storytelling, check out this [awesome walkthrough](https://tlp-lpa.ca/digital-skills/digital-storytelling) from which these tips have been adapted

1. **Make sure of conceptualizing your story using the dramatic arc**

<!-- Codes by HTML.am -->

<!-- CSS Code -->
<style type="text/css" scoped>
img.GeneratedImage {
width:400px;margin:10px;border-width:6px;border-color:#000000;border-style:solid;
}
</style>

<!-- HTML Code -->
<a href="https://www.researchgate.net/profile/Sara-Elshafie/publication/326720118/figure/fig1/AS:926468729208833@1597898766646/Freytags-pyramid-also-known-as-the-dramatic-arc-showing-a-five-part-story.png" target="_self"><img src="https://www.researchgate.net/profile/Sara-Elshafie/publication/326720118/figure/fig1/AS:926468729208833@1597898766646/Freytags-pyramid-also-known-as-the-dramatic-arc-showing-a-five-part-story.png" alt="Making Science Meaningful for Broad Audiences through Stories" class="Image" title="Wikipedia page for the CSS language"></a>

2. **Narrate your story in your own voice.** 

3. **Use different types of media like images, videos, and narration**

4. **Choose a tool, that helps you creating great stories**

#### Jupyter Books

One amazing way to create `living documents` is `Jupyter Book`.

Jupyter Books provides a flexible and powerful platform for creating, publishing, and sharing your work. Additionally, Jupyter Book allows you to combine `live code, equations, visualizations, and narrative text` in a single document. This makes it easy to create `engaging, interactive educational content` that helps students understand complex concepts.

Creating content using Jupyter Books is a simple process and we have created a [tutorial on how to use Jupyter Books](https://felixkoerber.github.io/jb/10min.html)!

```{dropdown} Tutorial on Jupyter Book

1. [Installing the prerequisites](https://felixkoerber.github.io/jb/setup.html)
Before you start setting up your course using Jupyter Book, make sure you have the following tools installed on your machine:
- **Git**: A version control system that helps you keep track of your code changes.
- **Jupyter Book**: A tool that helps you build and publish interactive books or documents. You can install Jupyter Book using pip install jupyter-book.
- A **text editor of your choice**: You can use any text editor, such as Visual Studio Code, Sublime Text, or Atom, to create and edit your content.


2. [Create a fresh Git Repository for your project](https://felixkoerber.github.io/jb/tutorialcontent/publishing/account.html#start-a-project-setup-a-public-repository)
2.1. Go to the GitHub website (https://github.com) and click on the plus button on the upper right corner.
![GHNewRepo](https://github.com/felixkoerber/jb/blob/main/static/New_repo.jpg?raw=true)

</br>

2.2. Create a new repository for your course by giving it a name and a description.

</br>

![GHNewRepo_Description](https://github.com/felixkoerber/jb/blob/main/static/new_repo_example.png?raw=true)

</br>

2.3. On GitHub, open up the empty project in your browser.
-  Navigate to `settings`, `actions`, then `general` to change **Workflow permissions** to **Read and Write Permission** to change the Workflow permissions to Read and Write Permission. This will allow you to push changes to the repository from your local machine.

</br>

![Workflow_Permissions](https://raw.githubusercontent.com/felixkoerber/jb/0bd9a2930a41bc3f79ad876b603ea5534ef1a23a/static/Workflow_permission.jpg)

</br>

2.4. Save your changes.

2.5. Open up a terminal window and navigate to the location where you want to store your local course copy.

2.6. Copy the project's link and clone the repository using the following command: git clone https://github.com/yourprojectname.
    
3. [Copy our course template](https://felixkoerber.github.io/jb/tutorialcontent/publishing/account.html#working-with-the-course-template)

- open our [course template repository](https://github.com/M-earnest/course_template_diler)

- click on `code` and then on *Download ZIP*
![Download_template](https://github.com/felixkoerber/jb/blob/main/static/Download_template.jpg?raw=true)

- Extract the contents of the ZIP file in the folder linked to your GitHub repository.
    
4. [Create Content](https://felixkoerber.github.io/jb/tutorialcontent/writing/writing.html)

- Open the Markdown (.md) or Jupyter (.ipynb) files and copy your interactive content and code.

- Make sure to give each file a meaningful name and add a title to each page.

- You can use the provided style guide as a reference to see how to effectively implement MyST Markdown.


5. [Table of Contents and config](https://felixkoerber.github.io/jb/tutorialcontent/structure.html)

-  Once you've created files, open the `_toc.yml`

-  add your newly created files in the sequence of your choice according to our template

-  open the `_config.yml`

-  Change the title, author, and the location of your GitHub repository.

6. [Share it online](https://felixkoerber.github.io/jb/tutorialcontent/publishing/account.html)

- In your terminal, navigate to the location of your project and type the following commands:

- a. ) `git add .`

- b. ) `git commit -m "my first commit`

- c. ) `git push`

- Alternatively use the Gitkraken Client


7. Add the pages

- On Github, navigate first to `settings` and then `pages`

- Click on `branch` and select `gh-pages`

![Set-Up_Pages](https://github.com/felixkoerber/jb/blob/main/static/Set_Up_Pages.jpg?raw=true)
You're all set! Once you're ready, make sure to make your repository public, so that others can view your beautiful website.

```

```{note}
GitHub Pages is a free web hosting service provided by GitHub. It allows one to easily publish websites directly from a GitHub repository. With GitHub Pages, you can create static websites, blogs, and even project portfolios without having to worry about managing servers or purchasing web hosting.

You simply create a repository in GitHub and push your website files (HTML, CSS, JavaScript, etc.) to the repository. GitHub Pages then automatically generates the website and makes it available at a unique URL. This URL is typically in the format "username.github.io/repository-name".

GitHub Pages makes it relatively easy to host your own website for developers and programmers, e.g. this course, but can be confusing for non-technical users. 
A workaround for non-technical users is to work with a preconfigured setup. Our workgroup for example has created a template that you can use to host a website like the one you're seeing right now. This is achieved by using a so called "github-worklflow". Checkout our tutorial for creating a website using Github pages [here](https://felixkoerber.github.io/jb/intro.html)

```

#### Closing Words

Overall, digital storytelling has become an important tool for educators looking to engage students in meaningful, creative projects. By providing a platform for students to explore and express their ideas in new and innovative ways, digital storytelling can help to promote empathy, understanding, and positive change in the classroom and beyond.