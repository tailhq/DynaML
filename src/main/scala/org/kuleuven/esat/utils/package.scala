package org.kuleuven.esat

import java.io.File

import com.github.tototoshi.csv.{QUOTE_NONNUMERIC, DefaultCSVFormat, CSVReader}

/**
 * Created by mandar on 4/2/15.
 */
package object utils {
  val log1pExp: (Double) => Double = (x) => {x + math.log1p(math.exp(-x))}

  def getCSVReader(file: String, delim: Char): CSVReader = {
    implicit object MyFormat extends DefaultCSVFormat {
      override val delimiter = delim
      override val quoting = QUOTE_NONNUMERIC
    }
    CSVReader.open(new File(file))
  }
}
