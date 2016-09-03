---
title: Miscelleneous Workflows
keywords: sample
sidebar: product1_sidebar
permalink: p1_sample5.html
tags: [pipes, workflow]
folder: product1
---


## General

### Duplicate a pipe

```scala
duplicate[S, D](pipe: DataPipe[S, D])
```

* _Type_: ```DataPipe[(S, S), (D, D)] ```
* _Result_: Takes a base pipe and creates a parallel pipe by duplicating it.

{% include links.html %}
