package org.kuleuven.esat

import java.io.File
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
}
