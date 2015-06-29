import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.zip.GZIPInputStream
import java.io.FileInputStream

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
    val iterator:BufferedReader = GzFileIterator(new java.io.File("data/SUSY.csv.gz"), "US-ASCII")
    (1 to args.toInt).foreach((_) =>println(iterator.readLine()))
    

  }

}