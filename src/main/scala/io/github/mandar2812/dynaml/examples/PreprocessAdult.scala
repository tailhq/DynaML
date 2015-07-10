package io.github.mandar2812.dynaml.examples

import java.io.BufferedReader
import com.github.tototoshi.csv.CSVWriter
import scala.util.Random

class ExtendedString(s:String) {
  def isNumber: Boolean = s.matches("[+-]?\\d+.?\\d+")
}

// and this is the companion object that provides the implicit conversion
object ExtendedString {
  implicit def String2ExtendedString(s:String): ExtendedString = new ExtendedString(s)
}

object PreprocessAdult {
  def apply(args: String = "") = {
    val iterator:BufferedReader = FileIterator(new java.io.File(args+"adult.data"),
      "US-ASCII")
    var line = iterator.readLine()
    val writer = CSVWriter.open(args+"adult.csv")
    val writert = CSVWriter.open(args+"adulttest.csv")

    println("Outputting train and test csv files ...")
    while(line != null) {

      val row = line.split(',').map(_.trim)
      val label = if(row.last == ">50K") 1.0 else -1.0
      val procrow = Array(row(0), row(2), row(4), row(10), row(11), row(12), label.toString)

      if(Random.nextDouble() <= 0.9)
      {
        writer.writeRow(procrow)
      } else {
        writert.writeRow(procrow)
      }
      line = iterator.readLine()
    }
    writer.close()
    writert.close()
    println("Done ...")
  }
}
