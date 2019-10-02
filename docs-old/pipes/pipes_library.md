---
title: Pipes Library
---

## DynaML Library Pipes

DynaML comes bundled with a set of data pipes which enable certain standard data processing tasks, they are defined in the ```DynaMLPipe``` object in the ```io.github.mandar2812.dynaml.pipes``` package and they can be invoked as ```DynaMLPipe.<pipe name>```.


------

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
//Import the workflow library.
import io.github.mandar2812.dynaml.DynaMLPipe._

val columns = List(0,3,5)
val dataPipe =
  fileToStream >
  replaceWhiteSpaces >
  extractTrainingFeatures(
    columns, Map(0 -> "NA", 3 -> "NA", 5 -> "NA")
  ) >
  removeMissingLines >
  streamToFile("processed_sample.csv")

val result = dataPipe("sample.csv")
```

Lets go over the code snippet piece by piece.

* First convert the text file to a Stream using ```fileToStream```
* Replace white spaces in each line by using ```replaceWhiteSpaces```
* Extract the required columns by ```extractTrainingFeatures```, be sure to supply it the column numbers (indexed from 0) and the missing value strings for each column to be extracted.
* Remove missing records ```removeMissingLines```
* Write the resulting data stream to a file ```streamToFile("processed_sample.csv")```

-----
