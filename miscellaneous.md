---
layout: post-index
permalink: /miscellaneous/
title: Miscellaneous
tagline: A List of Miscellaneous
tags: [miscellaneous]
comments: false
---

<!--
### [MOOC Achievements](https://leimao.github.io/miscellaneous/mooc-certificates/)

My archieve of MOOC achievements.

### [Useful Tools](https://leimao.github.io/miscellaneous/tools/)

Useful tools online.

### [Followings](https://leimao.github.io/miscellaneous/followings/)

The blogs I am following.
-->



{% for post in site.categories.miscellaneous %}
<!--
  {% capture year %}{{ post.date | date: '%Y' }}{% endcapture %}
  {% if year != year_previous %}
  <h2>{{ post.date | date: '%Y' }}</h2>
  {% endif %}
  {% capture year_previous %}{{ post.date | date: '%Y' }}{% endcapture %}
-->
  <h3><a href="{{ site.url }}{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a></h3>
  <p><i><small>Posted: {{ post.date | date: "%B %-d, %Y"}} Updated: {{ post.modified | date: "%B %-d, %Y"}}</small></i></p>
  {% if post.image.teaser %}
  <figure>
    <a href="{{ site.url }}{{ post.url }}"><img src="{{ site.url }}{{ post.image.teaser }}"></a>
  </figure>
  {% endif %}
  <p>{{ post.excerpt | strip_html | truncate: 160 }}</p>

{% endfor %}

