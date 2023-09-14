---
layout: post
title:  "딥러닝 세미나"
comments: true
---

딥러닝 관련 내외부 세미나를 공유하고자 합니다.

<div class="home">
  <ul class="post-list">
    {% for post in site.categories.seminar %}
      <li>
          <h3>
            <a class="post-link" href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
          </h3>
          <span class="post-meta">{{ post.date | date: "%b %-d, %Y %H:%m" }}</span>
          <hr id="line">
          <div class="content">
            {{ post.excerpt }}
            <a class="post-link" href="{{ post.url | prepend: site.baseurl }}"><img src="{{ post.image }}" style="max-width: 100%;height: auto;width: auto\9;"></a>
          </div>
        <br>
      </li>
    {% endfor %}
  </ul>
</div>