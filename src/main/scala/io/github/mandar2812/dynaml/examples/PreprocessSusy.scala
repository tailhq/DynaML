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
    while(line != null || line != "\n") {

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