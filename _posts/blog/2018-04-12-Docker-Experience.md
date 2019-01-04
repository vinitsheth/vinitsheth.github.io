---
layout: post
title: "Docker Beginner's Guide"
excerpt: "My Docker Experience as a Newbie"
modified: 2018-04-12T14:17:25-04:00
categories: blog
tags: [Software Engineering]
comments: true
share: true
---

### Introduction

Docker has become widely used in companies and education institutions for its convenience and portability compared to virtual machines. I was actually "forced" to used Docker and honestly I hated to use it probably because of the steepness at the beginning of the learning curve.

<br />

Here I documented some of the problems I met during my work and the potential solutions to solve them.

<div class = "titled-image">
<figure class = "titled-image">
    <img src = "{{ site.url }}/images/blog/2018-04-12-Docker-Experience/laurel-docker-containers.png">
    <figcaption>Carefully Working with Docker</figcaption>
</figure>
</div>



### Use Docker As App vs Virtual Machine

I first learned how to use Docker from the official tutorials on the Docker website. It basically taught you how to set up a portable web application in Docker. For developed software shippment, it is perfectly fine to use Docker as an App. However, in terms of software development, which requires debugging and some auxilliary components other than the software itself, using Docker as a Virtual Machine might be better.

<br />

When I say use Docker as an app, I mean you can just run the docker and wait for the response as a software program. The Docker container has been preset with ``CMD``, and all some part of your auxilliary files might have already been copied to the Docker container. Because the Docker container has been set with ``CMD``, it will run instantly just like a software program.

```bash
$ docker run octave-image-processing
```

When I say use Docker as a virtual machine, I mean you develop software interactively in Docker as if you are in a computer system. ``-it`` is used for interactive communication with the Docker container.

```bash
$ docker run -it octave-image-processing
```

### Mount Local Directories to Docker Container


Docker argument ``-v`` for running container allows you to bound the local directories to the directories in the Docker container. This facilitates the data streaming between local computer and Docker container.

```bash
$ docker run -it -v /mnt/data/leimao:/mnt -v /home/leimao:/home octave-image-processing
```

It should be noted that sometimes you do not have writting permissions to the local directories. In my case, I have two workstations, workstation ``alice`` and ``burning``, with my login name ``marine``. The ``marine`` account has both reading and writting to both of the work stations. One of the hard drive of ``alice`` was mounted to ``burning``. In ``burning``, the ``marine`` account could also read and write on the hard drive of ``alice`` that is mounted on the ``burning``. However, ``alice`` does not allow an account named ``root`` to write on the hard drive remotely. So in the Docker container of ``burning``, I do not have writting permission to the hard drive of ``alice`` mounted on ``burning`` because ``alice`` sees me as ``root``, instead of ``marine``, on ``burning``. In the Docker container of ``alice``, I can write to the hard drive because ``alice`` know this ``root`` is from ``marine`` on the local machine.


### Clean "Trash" Docker Images

Sometimes, you want to clean some Docker images to free more disk space. 

<br />

To check all the images created on the computer:

```bash
$ docker images
REPOSITORY                        TAG                   IMAGE ID            CREATED             SIZE
octave-image-processing           latest                9296aabfaedd        3 hours ago         1.16GB
slin/workspace                    latest                9216bcec9750        21 hours ago        5.37GB
python                            3.6.4                 07d72c0beb99        2 weeks ago         689MB
ubuntu                            16.04                 f975c5035748        5 weeks ago         112MB
nvidia/cuda                       9.0-cudnn7-devel      e2e0782eb2d9        5 weeks ago         2.86GB
nvidia/cuda                       9.1-cudnn7-devel      bfeba741488f        5 weeks ago         2.86GB
nvidia/cuda                       latest                f1cf7a94fb8b        5 weeks ago         2.23GB
tensorflow/tensorflow             latest                414b6e39764a        5 weeks ago         1.27GB
tensorflow/tensorflow             latest-gpu            aebd66be3e22        5 weeks ago         2.95GB
gcr.io/tensorflow/tensorflow      latest-gpu            aebd66be3e22        5 weeks ago         2.95GB
ubuntu                            latest                0458a4468cbc        2 months ago        112MB
ubuntu                            14.04                 dc4491992653        2 months ago        222MB
```

For example, to delete ``octave-image-processing`` created 3 hours ago using its ``IMAGE ID``:

```bash
$ docker rmi --force 9296aabfaedd
```

We use ``--force`` because sometimes the container is still running on this image.

<br />

Sometimes, you will find some images with name ``<none>`` that were images resulted of building failure. To remove them:

```bash 
$ docker rmi $(docker images --filter "dangling=true" -q --no-trunc)
```

### Choice of Base Images

To build your own images, we have to use some base images from the [Docker official repository](https://hub.docker.com/explore/) or some third-party repositories.

<br />

Sometimes, to save the disk space, we would use the base image with minimum size that is necessary for the enviroment. However, this sometimes will bring troubles.

<br />

For example, I had once want to install Python 3 and the latest Octave in the environment. I somehow chose a Python 3 image as my base image. Although I could still do ``apt-get update`` and ``apt-get install`` to install Octave, but the Octave installed would be a very old version. The personal package archives (PPA) which usually provides the latest version of Octave would also be problematic to use. Later, I changed my base image from Python 3 to Ubuntu 16.04, I was able to use PPA to install the latest Octave without any problem. So I guess many utilities used in Ubuntu were missing in the Python 3 base images.

### ARG vs ENV

When writting ``Dockerfile``, sometime you might be confused between ``ARG`` and ``ENV``. This figure from [Vladislav Supalov](https://vsupalov.com/docker-env-vars/) explained the difference very well.

<div class = "titled-image">
<figure class = "titled-image">
    <img src = "{{ site.url }}/images/blog/2018-04-12-Docker-Experience/arg_vs_env.png">
    <figcaption>ARG vs ENV in Docker</figcaption>
</figure>
</div>
