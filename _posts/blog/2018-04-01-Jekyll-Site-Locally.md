---
layout: post
title: "Setting Up Jekyll Site Locally"
excerpt: "Some Jekyll Notes"
modified: 2018-04-01T14:17:25-04:00
categories: blog
tags: [Web Development]
comments: true
share: true
---

### Introduction

Two years ago, I firstly learned Git, GitHub and built my first Jekyll site hosted on GitHub on Windows. However, because Jekyll does not support Windows without using a virtual machine, I had to upload new materials of my sites to GitHub and watch if the formatting and content were shown correctly on the site every time I wanted to update my site.

<br />

Since 2017, I have switched my working enviroment almost entirely to Ubuntu, but I still have not get Jekyll installed on my computer and tested my site locally, which is pretty bad for a person who studies computer science. Today, somehow I get a chance to get Jekyll installed and successfully hosted my site locally, which I should have done two years ago. Here I documented the process and caveates for my future references.

### Jekyll Installation

To properly install Jekyll, the following software and packages are necessary. 

```bash
$ sudo apt-get update
$ sudo apt-get install ruby ruby-dev make gcc
$ sudo gem install jekyll bundler
$ sudo apt-get install nodejs
$ bundle install
```

### Host Jekyll Site Locally

Go to the directory of the Jekyll site.

<br />

In ``_config.yml``, leave ``url`` blank.

```
# url: https://leimao.github.io # Host Jekyll site on GitHub
url: # Leave blank. Host Jekyll site locally on http://127.0.0.1:4000/ by default.
```

Start Jekyll server:

```bash
$ bundle exec jekyll serve
```

The following message will show if the server started succesfully.

```
Configuration file: /home/marine/Workspace/leimao.github.io/_config.yml
            Source: /home/marine/Workspace/leimao.github.io
       Destination: /home/marine/Workspace/leimao.github.io/_site
      Generating... 
                    done.
 Auto-regeneration: enabled for '/home/marine/Workspace/leimao.github.io'
Configuration file: /home/marine/Workspace/leimao.github.io/_config.yml
    Server address: http://127.0.0.1:4000/
  Server running... press ctrl-c to stop.
```

Now go to ``http://127.0.0.1:4000/`` in web browser to view the site.

<br />

Changing the local host url is feasible. But I am not going to learn how to do it for now.

### Inconsistencies

So far, everything looks OK except that the coding block does not show properly in the local host. Someone has also reflected similar [problems](https://github.com/MDAnalysis/MDAnalysis.github.io/issues/21) on GitHub. But I have not found solution to solve this problem.