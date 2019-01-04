---
layout: post
title: "TensorBoard on Docker"
excerpt: "Visualization of Tensorboard on Docker"
modified: 2018-05-15T14:17:25-04:00
categories: blog
tags: [TensorFlow, TensorBoard, Docker]
comments: true
share: true
---

### Introduction

TensorBoard has been developed by Google in order to accelarate the debugging process of TensorFlow and visualize the training process. However, I actually have not got a chance to use this tool until very recently. This blog documented the most basic protocols of visualizing TensorBoard running on the Docker on the remote sever from your local computer.

### Protocols

#### Add TensorBoard Components to Code

TensorBoard does not run on its own. You would need to add new components to your TensorFlow code to ask TensorBoard keep tracking of the tensor values.

<br />

Please check the official [TensorBoard Tutorial](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) about how to add such components.

#### Connect Ports of Docker Container to Server

This is usually done via the ``-p`` argument of ``docker run`` command. TensorBoard uses port ``6006`` by default, so we connect the port ``6006`` (``0.0.0.0:6006``) on Docker container to the port ``5001`` (``0.0.0.0:5001``) on the sever.

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
$ ssh -L 127.0.0.1:16006:0.0.0.0:5001 username@server
```
We use the full port name because sometimes there are warnings from the server terminal if we do not do so.

#### Restart Docker Container

To restart the docker container, in our server terminal:
```bash
$ docker start -i 05ee0d5a5a0e 
```
#### Start TensorBoard Service

To start the tensorboard service, in our docker container terminal:

```bash
$ tensorboard --logdir ./graphs/rnn/ &
```
We have to specifiy the TensorBoard record directories in the ``logdir`` argument. There might be remaining TensorBoard records in the directory. It would be better to clean the directory before getting new records from the new training, if we would like to monitor the new training process.

<br />

The ``&`` sign is used to run TensorBoard in the background. 

<br />

After starting TensorBoard successfully, we will receive such message:

```bash
TensorBoard 1.8.0 at http://05ee0d5a5a0e:6006 (Press CTRL+C to quit)
```

To kill TensorBoard process in the background if necessary, we first check the PID (Process ID) of Tensorboard using ``top``:

```bash
$ top
top - 16:24:57 up 69 days, 23:50,  0 users,  load average: 0.01, 0.29, 0.50
Tasks:   3 total,   1 running,   2 sleeping,   0 stopped,   0 zombie
%Cpu(s):  0.0 us,  0.1 sy,  0.0 ni, 99.9 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st
KiB Mem : 63850140 total, 11136512 free,  2106536 used, 50607092 buff/cache
KiB Swap:  4194300 total,  4191240 free,     3060 used. 60739944 avail Mem 

  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND    
    1 root      20   0   18232   2016   1528 S   0.0  0.0   0:00.07 bash 
   56 root      20   0 87.717g 603244 220008 S   0.0  0.9   0:14.92 tensorboard 
   98 root      20   0   38752   1808   1284 R   0.0  0.0   0:00.01 top 
```
In our case, the PID of Tensorboard is ``56``. We could kill the process in terminal:

```bash
$ kill -09 56
```

#### Run TensorFlow

Run the TensorFlow program in Docker container termnal:

```bash
$ python main.py
```

Use ``Ctrl`` + ``P`` + ``Ctrl`` + ``Q`` to exit the docker container while keep the container running in the backgroud if necessary.

#### Monitor TensorBoard Locally

Open web browser, such as Chrome, and go to the url ``http://127.0.0.1:16006``. We could see the TensorBoard and keep track of the training process on the remote server on our local computer.

