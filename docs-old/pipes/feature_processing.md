---
title: Transformations of Features
---

## Feature Processing


### Extract features and targets

```scala
splitFeaturesAndTargets
```

* _Type_: `#!scala DataPipe[Stream[String], Stream[(DenseVector[Double], Double)]]`
* _Result_: Take each line which is a comma separated string and extract all but the last element into a feature vector and leave the last element as the "target" value.


### Extract Specific Columns

```scala
extractTrainingFeatures(columns, missingVals)
```

* _Type_: `#!scala DataPipe[Stream[String], Stream[String]]`
* _Result_: Extract a subset of columns from a stream of comma separated string also replace any missing value strings with the empty string.
* _Usage_: `#!scala DynaMLPipe.extractTrainingFeatures(List(1,2,3), Map(1 -> "N.A.", 2 -> "NA", 3 -> "na"))`


### Gaussian Scaling of Data

```scala
gaussianScaling
```

* _Result_:  Perform gaussian normalization of features & targets, on a data stream which is a of the form `#!scala Stream[(DenseVector[Double], Double)]`.


### Gaussian Scaling of Train/Test Splits

```scala
gaussianScalingTrainTest
```

* _Result_:  Perform gaussian normalization of features & targets, on a data stream which is a `#!scala Tuple2` of the form `#!scala (Stream(training data), Stream(test data))`.


### Min-Max Scaling of Data

```scala
minMaxScaling
```

* _Result_:  Perform 0-1 scaling of features & targets, on a data stream which is a of the form `#!scala Stream[(DenseVector[Double], Double)]`.

### Min-Max Scaling of Train/Test Splits

```scala
minMaxScalingTrainTest
```

* _Result_:  Perform 0-1 scaling of features & targets, on a data stream which is a `#!scala Tuple2` of the form `#!scala (Stream(training_data), Stream(test_data))`.
