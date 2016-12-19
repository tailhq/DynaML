/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
* */
package io.github.mandar2812.dynaml

import java.io.{BufferedWriter, File, FileWriter}

import breeze.linalg.{DenseMatrix, DenseVector, Matrix, MatrixNotSquareException, MatrixNotSymmetricException, kron}
import com.github.tototoshi.csv.{CSVReader, DefaultCSVFormat, QUOTE_NONNUMERIC}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.io.Source
import scala.reflect.runtime.{universe => ru}
import scala.annotation.tailrec
import scala.util.matching.Regex
import sys.process._
import java.net.URL
import scalaxy.streams.optimize
import spire.algebra.Field

/**
  * A set of pre-processing utilities
  * and library functions.
  */
package object utils {
  val log1pExp: (Double) => Double = (x) => {x + math.log1p(math.exp(-x))}
  /**
    * Get a [[CSVReader]] object from a file name and delimiter
    * character.
    *
    * @param file The file pathname as a String
    * @param delim The delimiter character used
    *              to split the csv file.
    * @return A [[CSVReader]] object which can be
    *         iterated for each line.
    * */
  def getCSVReader(file: String, delim: Char): CSVReader = {
    implicit object MyFormat extends DefaultCSVFormat {
      override val delimiter = delim
      override val quoting = QUOTE_NONNUMERIC
    }
    CSVReader.open(new File(file))
  }

  /**
    * Get the mean and variance of a data set
    * which is a [[List]] of [[DenseVector]].
    *
    * @param data The data set.
    * @return A [[Tuple2]] containing the mean
    *         and variance * n-1.
    *
    * */
  def getStats(data: List[DenseVector[Double]]):
  (DenseVector[Double], DenseVector[Double]) = {
    @tailrec
    def getStatsRec(d: List[DenseVector[Double]],
                    m: DenseVector[Double],
                    s: DenseVector[Double],
                    i: Int):
    (DenseVector[Double], DenseVector[Double]) = d match {
      case Nil => (m, s)
      case x :: rest =>
        getStatsRec(rest, m + (x - m)/i.toDouble,
          s + ((x - m) :* (x - (m + (x - m)/i.toDouble))),
          i - 1)

    }

    getStatsRec(data.tail, data.head,
      DenseVector.zeros[Double](data.head.length),
      data.length)
  }


  def getMinMax(data: List[DenseVector[Double]]):
  (DenseVector[Double], DenseVector[Double]) = {
    @tailrec
    def getMinMaxRec(d: List[DenseVector[Double]],
                     m: DenseVector[Double],
                     s: DenseVector[Double],
                     i: Int):
    (DenseVector[Double], DenseVector[Double]) = d match {
      case Nil => (m, s)
      case x :: rest =>
        getMinMaxRec(rest,
          DenseVector((x.toArray zip m.toArray).map(c => math.min(c._1, c._2))),
          DenseVector((x.toArray zip s.toArray).map(c => math.max(c._1, c._2))),
          i - 1)

    }

    getMinMaxRec(
      data.tail,
      data.head,
      data.head,
      data.length)
  }

  /*
  * Calculate the value of the hermite polynomials
  * tail recursively. This is needed to calculate
  * the Gaussian derivatives at a point x.
  * */
  def hermite(n: Int, x: Double): Double = {
    @tailrec
    def hermiteHelper(k: Int, x: Double, a: Double, b: Double): Double =
      k match {
        case 0 => a
        case 1 => b
        case _ => hermiteHelper(k-1, x, b, x*b - (k-1)*a)
      }
    hermiteHelper(n, x, 1, x)
  }

  @tailrec
  def factorial(n: Int, accumulator: Long = 1): Long = {
    if(n == 0) accumulator else factorial(n - 1, (accumulator * n))
  }

  /**
    * Get the mean and variance of a data set
    * which is a [[List]] of [[DenseVector]].
    *
    * @param data The data set.
    * @return A [[Tuple2]] containing the mean
    *         and variance.
    *
    * */

  def getStatsMult(data: List[DenseVector[Double]]):
  (DenseVector[Double], DenseMatrix[Double]) = {
    def getStatsRec(d: List[DenseVector[Double]],
                    m: DenseVector[Double],
                    s: DenseMatrix[Double],
                    i: Int):
    (DenseVector[Double], DenseMatrix[Double]) = d match {
      case Nil =>
        m :/= i.toDouble
        s :/= i.toDouble
        //val m1: DenseVector[Double] = m/i.toDouble
        (m, s - (m*m.t))

      case x :: rest =>
        getStatsRec(rest, m + x,
          s + x*x.t,
          i + 1)

    }

    getStatsRec(data.tail, data.head,
      data.head * data.head.t,
      1)
  }

  def getStatsRDD(data: RDD[LabeledPoint]):
  (Double, Double,
    DenseVector[Double],
    DenseMatrix[Double]) = {
    val (lm, ls, m, s) = data.map((p) => {
      val label = p.label
      val features = DenseVector(p.features.toArray)
      (label, label*label, features, features*features.t)
    }).reduce((a,b) => {
      (a._1 + b._1, a._2 + b._2, a._3 + b._3, a._4 + b._4)
    })
    val count = data.count().toDouble
    val labelMean = lm/count
    val labelVar = (ls/count) - labelMean*labelMean
    m :/= count
    s :/= count
    val featuresCov = s - m*m.t

    (labelMean, labelVar, m, featuresCov)
  }

  def getTypeTag[T: ru.TypeTag](obj: T) = ru.typeTag[T]

  def combine[A](xs: Traversable[Traversable[A]]): Seq[Seq[A]] = optimize {
    xs.foldLeft(Seq(Seq.empty[A])) {
      (x, y) => for (a <- x.view; b <- y) yield a :+ b
    }
  }

  def downloadURL(url: String, saveAs: String): Unit =
    new URL(url) #> new File(saveAs) !!

  def replace(find: String)(replace: String)(input: String): String = {
    val pattern = new Regex(find)
    pattern.replaceAllIn(input, replace)
  }

  def textFileToStream(fileName: String): Stream[String] =
    Source.fromFile(new File(fileName)).getLines().toStream

  def strReplace(fileName: String)
                (findStringRegex: String, replaceString: String)
  : Stream[String] = optimize {textFileToStream(fileName)
    .map(replace(findStringRegex)(replaceString))}

  def writeToFile(destination: String)(lines: Stream[String]): Unit = {
    val writer = new BufferedWriter(new FileWriter(new File(destination)))
    lines.foreach(line => {
      writer.write(line+"\n")
    })
    writer.close()
  }

  def transformData(transform: (String) => String)(lines: Stream[String]): Stream[String] =
    optimize { lines.map(transform) }

  def extractColumns(lines: Stream[String], sep: String,
                     columns: List[Int], naStrings:Map[Int, String]): Stream[String] = {
    val tFunc = (line: String) => {
      val fields = line.split(sep)

      optimize {
        val newFields:List[String] = columns.map(col => {
          if (!naStrings.contains(col) || fields(col) != naStrings(col)) fields(col)
          else "<NA>"
        })

        val newLine = newFields.foldLeft("")(
          (str1, str2) => str1+sep+str2
        )

        newLine.tail
      }
    }

    transformData(tFunc)(lines)
  }

  /**
    * Construct a haar transform matrix of size n
    *
    * NOTE: n must be a power of 2.
    *
    * */
  def haarMatrix(n: Int) = {
    val pos = DenseMatrix(Array(1.0, 1.0))
    val neg = DenseMatrix(Array(-1.0, 1.0))
    val hMat = DenseMatrix(Array(1.0, 1.0), Array(-1.0, 1.0))
    def haarMatrixAcc(i: Int, hMatAcc: DenseMatrix[Double]): DenseMatrix[Double] = i match {
      case `n` => hMatAcc
      case index =>
        haarMatrixAcc(i*2,
          DenseMatrix.vertcat[Double](
            kron(hMatAcc, pos),
            kron(DenseMatrix.eye[Double](i), neg)))
    }
    haarMatrixAcc(2, hMat)
  }

  def productField[Domain, Domain1](ev: Field[Domain], ev1: Field[Domain1]): Field[(Domain, Domain1)] =
    new Field[(Domain, Domain1)] {
      override def gcd(a: (Domain, Domain1), b: (Domain, Domain1)): (Domain, Domain1) =
        (ev.gcd(a._1, b._1), ev1.gcd(a._2, b._2))

      override def quot(a: (Domain, Domain1), b: (Domain, Domain1)): (Domain, Domain1) =
        (ev.quot(a._1, b._1), ev1.quot(a._2, b._2))

      override def mod(a: (Domain, Domain1), b: (Domain, Domain1)): (Domain, Domain1) =
        (ev.mod(a._1, b._1), ev1.mod(a._2, b._2))

      override def negate(x: (Domain, Domain1)): (Domain, Domain1) =
        (ev.negate(x._1), ev1.negate(x._2))

      override def zero: (Domain, Domain1) = (ev.zero, ev1.zero)

      override def one: (Domain, Domain1) = (ev.one, ev1.one)

      override def plus(x: (Domain, Domain1), y: (Domain, Domain1)): (Domain, Domain1) =
        (ev.plus(x._1, y._1), ev1.plus(x._2, y._2))

      override def div(x: (Domain, Domain1), y: (Domain, Domain1)): (Domain, Domain1) =
        (ev.div(x._1, y._1), ev1.div(x._2, y._2))

      override def times(x: (Domain, Domain1), y: (Domain, Domain1)): (Domain, Domain1) =
        (ev.times(x._1, y._1), ev1.times(x._2, y._2))
    }


  def isSquareMatrix[V](mat: Matrix[V]): Unit =
    if (mat.rows != mat.cols)
      throw new MatrixNotSquareException

  def isSymmetricMatrix[V](mat: Matrix[V]): Unit = {
    isSquareMatrix(mat)

    optimize {
      for (i <- 0 until mat.rows; j <- 0 until i)
        if (mat(i,j) != mat(j,i))
          throw new MatrixNotSymmetricException
    }
  }

}

