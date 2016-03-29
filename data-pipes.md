---
layout: page
title: Data Pipes
---

-----

## Configurable Data Workflows
Data _munging_ or pre-processing is one of the most time consuming activities in the analysis and modeling cycle, yet very few libraries do justice to this need. In DynaML the aim has been to make data analysis more reproducible and easy, hence designing, maintaining and improving a powerful data workflow framework is at the center of the development endeavour. In this section we attempt to give a simple yet effective introduction to the data pipes module of DynaML.

## What are DynaML Data Pipes?

At their heart data pipes in DynaML are (thinly wrapped) Scala functions. Every pre-processing workflow can be visualized as a chain of functional transformations on the data. These functional transformations are applied one after another (in fancy language _composed_) to yield a result which is then suitable for modeling/training.


### Creating an arbitrary pipe

As we mentioned earlier a DynaML pipe is nothing but a thin wrapper around a scala function. Creating a new data pipe is very easy, you just create a scala function and give it to the ```DataPipe()``` object.

### Joining Data Pipes

You can compose or join any number of pipes using the ```>``` character to create a composite data workflow. There is only one constraint when joining two pipes, that the destination type of the first pipe must be the same as the source type of the second pipe, in other words "dont put square pegs into round holes".

-----


# DynaML Library Pipes

DynaML comes bundled with a set of data pipes which enable certain standard data processing tasks.


Data Pre-processing
------


### ```fileToStream```

* _Type_: ```DataPipe[String, Stream[String]]```
* _Result_: Converts a text file (inputted as a file path string) into ```Stream[String]```   


### ```streamToFile(fileName: String)```

* _Type_: ```DataPipe[Stream[String], Unit] ```
* _Result_: Writes a stream of lines to the file specified by ```filePath```


### ```dropHead```

* _Type_: ```DataPipe[Stream[String], Stream[String]] ```
* _Result_: Drop the first element of a ```Stream``` of ```String```


### ```replace(original, newString)```

* _Type_: ```DataPipe[Stream[String], Stream[String]] ```
* _Result_: Replace all occurrences of a regular expression or string in a ```Stream``` of ```String``` with with a specified replacement string.


### ```replaceWhiteSpaces```

* _Type_: ```DataPipe[Stream[String], Stream[String]] ```
* _Result_: Replace all white space characters in a stream of lines.


### ```trimLines```

* _Type_: ```DataPipe[Stream[String], Stream[String]] ```
* _Result_: Trim white spaces from both sides of every line.


### ```removeMissingLines```

* _Type_: ```DataPipe[Stream[String], Stream[String]] ```
* _Result_: Remove all lines/records which contain missing values


### ```replaceWhiteSpaces```

* _Type_: ```DataPipe[Stream[String], Stream[String]] ```
* _Result_: Replace all white space characters in a stream of lines.


### ```splitTrainingTest(num_training, num_test)```

* _Type_: ```DataPipe[(Stream[(DenseVector[Double], Double)], Stream[(DenseVector[Double], Double)]),
(Stream[(DenseVector[Double], Double)], Stream[(DenseVector[Double], Double)])] ```
* _Result_: Extract a subset of the data into a ```Tuple2``` which can be used as a training, test combo for model learning and evaluation.



Feature Processing
------


### ```splitFeaturesAndTargets```

* _Type_: ```DataPipe[Stream[String], Stream[(DenseVector[Double], Double)]] ```
* _Result_: Take each line which is a comma separated string and extract all but the last element into a feature vector and leave the last element as the "target" value.


### ```extractTrainingFeatures(columns, missingVals)```

* _Type_: ```DataPipe[Stream[String], Stream[String]] ```
* _Result_: Extract a subset of columns from a stream of comma separated string also replace any missing value strings with the empty string.
* _Usage_: ```DynaMLPipe.extractTrainingFeatures(List(1,2,3), Map(1 -> "N.A.", 2 -> "NA", 3 -> "na"))```


### ```trainTestGaussianStandardization```

* _Result_:  Perform gaussian normalization on a data stream which is a ```Tuple2``` of the form `(Stream(training data), Stream(test data))`.



Time Series Data
------


### ```extractTimeSeries(Tfunc)```

* _Type_: ```DataPipe[Stream[String], Stream[(Double, Double)]] ```
* _Result_: This pipe assumes its input to be of the form `YYYY,Day,Hour,Value`. It takes as input a function (TFunc) which converts a ```(Double, Double, Double)``` into a single "timestamp" like value. The pipe processes its data source line by line and outputs a ```Tuple2``` in the following format `(Timestamp,Value)`.


### ```deltaOperation(deltaT, timelag)```

* _Type_: ```DataPipe[Stream[(Double, Double)], Stream[(DenseVector[Double], Double)]] ```
* _Result_: Inorder to generate features for auto-regressive models, one needs to construct sliding windows in time. This function takes two parameters `deltaT`: the auto-regressive order and `timelag`: the time lag after which the windowing is conducted. E.g Let deltaT = 2 and timelag = 1 This pipe will take stream data of the form $$(t, y(t))$$ and output a stream which looks like $$(t, [y(t-2), y(t-3)])$$


### ```extractTimeSeriesVec(Tfunc)```

* _Type_: ```DataPipe[Stream[String], Stream[(Double, DenseVector[Double])]] ```
* _Result_: This pipe is similar to ```extractTimeSeries``` but for application in multivariate time series analysis such as nonlinear autoregressive models with exogenous inputs. The pipe processes its data source line by line and outputs a ```(Double, DenseVector[Double])``` in the following format `(Timestamp,Values)`.


### ```deltaOperationVec(deltaT: Int)```

* _Type_: ```DataPipe[Stream[(Double, Double)], Stream[(DenseVector[Double], Double)]] ```
* _Result_: A variant of `deltaOperation` for NARX models.



General
------

### ```duplicate[S, D](pipe: DataPipe[S, D])```

* _Type_: ```DataPipe[(S, S), (D, D)] ```
* _Result_: Takes a base pipe and creates a parallel pipe by duplicating it.


-----

## Example
As a simple motivating example consider the following hypothetical csv data file called ```sample.csv```.

```
a  b  c  NA  e f
r  s  q  t  l   m
u v w x z d
```

Lets say one wants to extract only the first, fourth and last columns of this file for further processing, also one is only interested in records which do not have missing values in any of the columns we want to extract. One can think of a data pipe as follows.

* Replace the erratic white space separators with a consistent separator character
* Extract a subset of the columns
* Remove the records with missing values ```NA```
* Write output to another file ```processedsample.csv``` with the comma character as separator

We can do this by 'composing' data flow pipes which achieve each of the sub tasks.

```scala
	val columns = List(0,3,5)
	val dataPipe =
		DynaMLPipe.fileToStream >
		DynaMLPipe.replaceWhiteSpaces >
		DynaMLPipe.extractTrainingFeatures(
			columns,
			Map(0 -> "NA", 3 -> "NA", 5 -> "NA")
		) >
		DynaMLPipe.removeMissingLines >
		DynaMLPipe.streamToFile("processedsample.csv")

	val result = dataPipe run "sample.csv"
```

Lets go over the code snippet piece by piece.

* First convert the text file to a Stream using ```DynaMLPipe.fileToStream```
* Replace white spaces in each line by using ```DynaMLPipe.replaceWhiteSpaces```
* Extract the required columns by ```DynaMLPipe.extractTrainingFeatures```, be sure to supply it the column numbers (indexed from 0) and the missing value strings for each column to be extracted.
* Remove missing records ```DynaMLPipe.removeMissingLines```
* Write the resulting data stream to a file ```DynaMLPipe.streamToFile("processedsample.csv")```

-----

## Useful Links

### DynaML API


* [DynaMLPipe](http://mandar2812.github.io/DynaML/target/site/scaladocs/index.html#io.github.mandar2812.dynaml.pipes.DynaMLPipe$): Bundled data pipes.
* [Pipes package](http://mandar2812.github.io/DynaML/target/site/scaladocs/index.html#io.github.mandar2812.dynaml.pipes.package)

