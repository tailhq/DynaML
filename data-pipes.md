---
layout: page
title: Data Pipes
---

## Configurable Data Workflows
Data _munging_ or pre-processing is one of the most time consuming activities in the analysis and modeling cycle, yet very few libraries do justice to this need. In DynaML the aim has been to make data analysis more reproducible and easy, hence designing, maintaining and improving a powerful data workflow framework is at the center of the development endeavour. In this section we attempt to give a simple yet effective introduction to the data pipes module of DynaML.

## What are DynaML Data Pipes?

At their heart data pipes in DynaML are (thinly wrapped) Scala functions. Every pre-processing workflow can be visualized as a chain of functional transformations on the data. These functional transformations are applied one after another (in fancy language _composed_) to yield a result which is then suitable for modeling/training.


### Creating an arbitrary pipe

As we mentioned earlier a DynaML pipe is nothing but a thin wrapper around a scala function. Creating a new data pipe is very easy, you just create a scala function and give it to the ```DataPipe()``` object.

### Joining Data Pipes

You can compose or join any number of pipes using the ```>``` character to create a composite data workflow. There is only one constraint when joining two pipes, that the destination type of the first pipe must be the same as the source type of the second pipe, in other words "dont put square pegs into round holes".

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
		DataPipe(utils.writeToFile("processedsample.csv") _)

	dataPipe.run("sample.csv")
```

Lets go over the code snippet piece by piece.

* First convert the text file to a Stream using ```DynaMLPipe.fileToStream```
* Replace white spaces in each line by using ```DynaMLPipe.replaceWhiteSpaces```
* Extract the required columns by ```DynaMLPipe.extractTrainingFeatures```, be sure to supply it the column numbers (indexed from 0) and the missing value strings for each column to be extracted.
* Remove missing records ```DynaMLPipe.removeMissingLines```
* Write the resulting data stream to a file ```DataPipe(utils.writeToFile("processedsample.csv") _)```

## Useful Links

### DynaML API

* [DynaML library pipes](http://mandar2812.github.io/DynaML/target/site/scaladocs/index.html#io.github.mandar2812.dynaml.pipes.DynaMLPipe$)

* [Pipes package](http://mandar2812.github.io/DynaML/target/site/scaladocs/index.html#io.github.mandar2812.dynaml.pipes.package)

