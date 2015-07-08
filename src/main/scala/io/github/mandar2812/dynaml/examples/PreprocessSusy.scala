package io.github.mandar2812.dynaml.examples

import java.io.{BufferedReader, FileInputStream, InputStreamReader}
import java.util.zip.GZIPInputStream

import com.github.tototoshi.csv.CSVWriter

import scala.util.Random

case class BufferedReaderIterator(reader: BufferedReader) extends Iterator[String] {
  override def hasNext() = reader.ready
  override def next() = reader.readLine()
}

object GzFileIterator {
  def apply(file: java.io.File, encoding: String): BufferedReader = {
    new BufferedReader(
      new InputStreamReader(
        new GZIPInputStream(
          new FileInputStream(file)), encoding))
  }
}

object PreprocessSusy {
  def apply(args: String = "") = {
    val iterator:BufferedReader = GzFileIterator(new java.io.File(args+"SUSY.csv.gz"),
      "US-ASCII")
    var line = iterator.readLine()
    val writer = CSVWriter.open(args+"susy.csv")
    val writert = CSVWriter.open(args+"susytest.csv")

    println("Outputting train and test csv files ...")
    while(line != null) {

      val row = line.split(',').reverse
      val procrow = Array.tabulate(row.length)((i) => {
        if(i == row.length-1) {
          val label = if(row(i).toDouble == 1.0) row(i).toDouble else -1.0
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