---
title: Miscelleneous Workflows
---


## General

### Duplicate a pipe

```scala
duplicate[S, D](pipe: DataPipe[S, D])
```

* _Type_: ```DataPipe[(S, S), (D, D)] ```
* _Result_: Takes a base pipe and creates a parallel pipe by duplicating it.
