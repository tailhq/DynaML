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

object PreprocessSusy {
  def apply(args: String = "") = {
    val iterator:BufferedReader = GzFileIterator(new java.io.File(args+"SUSY.csv.gz"),
      "US-ASCII")
    var line = iterator.readLine()
    val writer = CSVWriter.open(args+"susy.csv")
    val writert = CSVWriter.open(args+"susytest.csv")

    println("Outputting train and test csv files ...")
    while(line.toBoolean) {
      if(Random.nextDouble() <= 0.9)
      {
        writer.writeRow(line.split(',').reverse)
      } else {
        writert.writeRow(line.split(',').reverse)
      }
      line = iterator.readLine()
    }
    writer.close()
    writert.close()
    println("Done ...")
  }
}