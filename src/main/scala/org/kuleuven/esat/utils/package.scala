package org.kuleuven.esat

import java.io.File
import breeze.linalg.DenseVector
import com.github.tototoshi.csv.{QUOTE_NONNUMERIC, DefaultCSVFormat, CSVReader}

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
   *
   * @return A [[Tuple2]] containing the mean
   *         and variance * n-1.
   *
   * */
  def getStats(data: List[DenseVector[Double]]):
  (DenseVector[Double], DenseVector[Double]) = {
    def getStatsRec(d: List[DenseVector[Double]],
                    m: DenseVector[Double],
                    s: DenseVector[Double],
                    i: Int):
    (DenseVector[Double], DenseVector[Double]) = d match {
      case Nil => (m, s)
      case x :: rest => {
        getStatsRec(rest, m + (x - m):/=i.toDouble,
          s + ((x - m) :* (x - (m + (x - m):/=i.toDouble))),
          i - 1)
      }
    }

    getStatsRec(data.tail, data.head,
      DenseVector.zeros[Double](data.head.length),
      data.length)
  }
}
