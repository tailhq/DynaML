---
title: Transform File Contents
keywords: file_pipes
summary: "A List of data pipes useful for processing contents of data files."
sidebar: product1_sidebar
tags: [pipes, workflow]
permalink: p1_sample2.html
folder: product1
---

## Data Pre-processing

### File to Stream of Lines

```scala
fileToStream
```

* _Type_: ```DataPipe[String, Stream[String]]```
* _Result_: Converts a text file (inputted as a file path string) into ```Stream[String]```   


### Write Stream of Lines to File

```scala
streamToFile(fileName: String)
```

* _Type_: ```DataPipe[Stream[String], Unit] ```
* _Result_: Writes a stream of lines to the file specified by ```filePath```


### Drop first line in Stream

```scala
dropHead
```

* _Type_: ```DataPipe[Stream[String], Stream[String]] ```
* _Result_: Drop the first element of a ```Stream``` of ```String```


### Replace Occurrences in of a String

```scala
replace(original, newString)
```

* _Type_: ```DataPipe[Stream[String], Stream[String]] ```
* _Result_: Replace all occurrences of a regular expression or string in a ```Stream``` of ```String``` with with a specified replacement string.


### Replace White Spaces

```scala
replaceWhiteSpaces
```

* _Type_: ```DataPipe[Stream[String], Stream[String]] ```
* _Result_: Replace all white space characters in a stream of lines.


### Remove Trailing White Spaces

* _Type_: ```DataPipe[Stream[String], Stream[String]] ```
* _Result_: Trim white spaces from both sides of every line.

### Remove White Spaces

```scala
replaceWhiteSpaces
```

* _Type_: ```DataPipe[Stream[String], Stream[String]] ```
* _Result_: Replace all white space characters in a stream of lines.

### Remove Missing Records

```scala
removeMissingLines
```

* _Type_: ```DataPipe[Stream[String], Stream[String]] ```
* _Result_: Remove all lines/records which contain missing values


### Create Train/Test splits

```scala
splitTrainingTest(num_training, num_test)
```

* _Type_: ```DataPipe[(Stream[(DenseVector[Double], Double)], Stream[(DenseVector[Double], Double)]),
(Stream[(DenseVector[Double], Double)], Stream[(DenseVector[Double], Double)])] ```
* _Result_: Extract a subset of the data into a ```Tuple2``` which can be used as a training, test combo for model learning and evaluation.

{% include links.html %}
