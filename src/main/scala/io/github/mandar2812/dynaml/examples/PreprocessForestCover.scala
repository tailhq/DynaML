package io.github.mandar2812.dynaml.examples

import java.io.{BufferedReader, FileInputStream, InputStreamReader}

import com.github.tototoshi.csv.CSVWriter

import scala.util.Random

object FileIterator {
  def apply(file: java.io.File, encoding: String): BufferedReader = {
    new BufferedReader(
      new InputStreamReader(
        new FileInputStream(file), encoding))
  }
}

object PreprocessForestCover {
  def apply(args: String = "") = {
    val iterator:BufferedReader = FileIterator(new java.io.File(args+"covtype.data"),
      "US-ASCII")
    var line = iterator.readLine()
    val writer = CSVWriter.open(args+"cover.csv")
    val writert = CSVWriter.open(args+"covertest.csv")

    println("Outputting train and test csv files ...")
    while(line != null) {

      val row = line.split(',')
      val procrow = Array.tabulate(row.length)((i) => {
        if(i == row.length-1) {
          val label = if(row(i).toDouble == 2.0) 1.0 else -1.0
          label.toString
        } else {
          row(i)
        }
      })

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