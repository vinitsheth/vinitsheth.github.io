---
layout: post
title: "Jupyter Notebook on Docker"
excerpt: "Using Jupyter Notebook on Docker"
modified: 2018-05-15T14:17:25-04:00
categories: blog
tags: [Jupyter Notebook, Docker]
comments: true
share: true
---

### Introduction

Although I do not like using Jupyter Notebook due to its "bad" garbage collection mechanism, I still sometimes have to use it on the Docker. This blog documented the most basic protocols of connecting to the Jupyter Notebook running on the Docker on the remote sever from your local computer.

### Protocols

#### Connect Ports of Docker Container to Server

This is usually done via the ``-p`` argument of ``docker run`` command. Jupyter Notebook uses port ``8888`` by default, so we connect the port ``8888`` (``0.0.0.0:8888``) on Docker container to the port ``5000`` (``0.0.0.0:5000``) on the sever.

```bash
$ nvidia-docker run -it --name leimao-speech-instance -v /home/leimao/workspace:/workspace -p 5000:8888 -p 5001:6006 leimao/speech
```
To exit the docker container while keep the container running in the backgroud, click ``Ctrl`` + ``P`` + ``Ctrl`` + ``Q``.

<br />

Run ``docker container ls`` to check if we have connected the port successfully.

```bash
$ docker container ls
CONTAINER ID        IMAGE                        COMMAND             CREATED             STATUS              PORTS                                              NAMES
05ee0d5a5a0e        leimao/speech                "/bin/bash"         About an hour ago   Up 8 seconds        0.0.0.0:5001->6006/tcp, 0.0.0.0:5000->8888/tcp     leimao-speech-instance
```


#### SSH to Server

To connect the local port to the server port, in our local terminal:

```bash
$ ssh -L 127.0.0.1:16006:0.0.0.0:5000 username@server
```
We use the full port name because sometimes there are warnings from the server terminal if we do not do so.

#### Restart Docker Container

To restart the docker container, in our server terminal:
```bash
$ docker start -i 05ee0d5a5a0e 
```
#### Start Jupyter Notebook

To start jupyter notebook, go to the ``workspace`` directory and use the following command in our docker container terminal:

```bash
$ jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root &
```
``--allow-root`` argument sometimes prevents weird errors.

<br />

The ``&`` sign is used to run Jupyter Notebook in the background. 

<br />

After starting Jupyter Notebook successfully, we will receive such message:

```bash
[I 13:46:05.790 NotebookApp] Serving notebooks from local directory: /
[I 13:46:05.790 NotebookApp] 0 active kernels
[I 13:46:05.790 NotebookApp] The Jupyter Notebook is running at:
[I 13:46:05.790 NotebookApp] http://aa01088dd584:8888/?token=7b3ffc8481d4a664f3d44dcb418af5a2492087711adf7649
[I 13:46:05.790 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 13:46:05.790 NotebookApp] No web browser found: could not locate runnable browser.
[C 13:46:05.790 NotebookApp] 
    
    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://aa01088dd584:8888/?token=7b3ffc8481d4a664f3d44dcb418af5a2492087711adf7649&token=7b3ffc8481d4a664f3d44dcb418af5a2492087711adf7649
```

Use ``Ctrl`` + ``P`` + ``Ctrl`` + ``Q`` to exit the docker container while keep the container running in the backgroud if necessary.

<br />

To kill Jupyter Notebook process in the background if necessary, we first check the PID (Process ID) of Jupyter Notebook using ``top``:

```bash
$ top
top - 16:24:57 up 69 days, 23:50,  0 users,  load average: 0.01, 0.29, 0.50
Tasks:   3 total,   1 running,   2 sleeping,   0 stopped,   0 zombie
%Cpu(s):  0.0 us,  0.1 sy,  0.0 ni, 99.9 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
KiB Mem : 63850140 total, 11136512 free,  2106536 used, 50607092 buff/cache
KiB Swap:  4194300 total,  4191240 free,     3060 used. 60739944 avail Mem 

  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND    
    1 root      20   0   18228   1976   1504 S   0.0  0.0   0:00.03 bash        
   49 root      20   0  325408  50080   8924 S   0.0  0.1   0:00.64 jupyter-no+ 
   56 root      20   0  598748  39508   7016 S   0.0  0.1   0:00.31 python      
   73 root      20   0   36628   1672   1248 R   0.0  0.0   0:00.01 top         
```
In our case, the PID of Jupyter Notebook is ``49``. We could kill the process in terminal:

```bash
$ kill -09 49
```

#### Use Jupyter Notebook Locally

Open web browser, such as Chrome, and go to the url ``http://127.0.0.1:16006``. Input token ``7b3ffc8481d4a664f3d44dcb418af5a2492087711adf7649`` as requested. Alternatively, use the url provided in the message we saw after starting the Jupyter Notebook for the first time connection and do not forget to replace ``http://aa01088dd584:8888`` to ``http://127.0.0.1:16006``.
