import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.zip.GZIPInputStream
import java.io.FileInputStream

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

object PreprocessForestCover {
  def apply(args: String = "") = {
    val iterator:BufferedReader = GzFileIterator(new java.io.File(args+"covtype.data"),
      "US-ASCII")
    var line = iterator.readLine()
    val writer = CSVWriter.open(args+"cover.csv")
    val writert = CSVWriter.open(args+"covertest.csv")

    println("Outputting train and test csv files ...")
    while(line != null) {

      val row = line.split(',')
      val procrow = Array.tabulate(row.length)((i) => {
        if(i == row.length-1) {
          val label = if(row(i).toDouble <= 4.0) 1.0 else -1.0
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