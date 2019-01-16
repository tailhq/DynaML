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
package io.github.mandar2812.dynaml.evaluation

import org.platanios.tensorflow.api._

import org.json4s._
import org.json4s.jackson.Serialization.{read => read_json, write => write_json}



/**
  * Top level class for metrics computed on Tensorflow objects.
  *
  * @param preds Predictions
  *
  * @param targets The actual output values.
  * */
abstract class MetricsTF(val names: Seq[String], val preds: Tensor, val targets: Tensor) {

  implicit val formats = DefaultFormats

  protected val scoresAndLabels: (Tensor, Tensor) = (preds, targets)

  protected var name = "Target"

  lazy val results: Tensor = run()

  def _target_quantity: String = name

  def target_quantity_(n: String): Unit = {
    name = n
  }

  def print(): Unit = {
    println("\nModel Performance: "+name)
    println("============================")
    println()

    names.zipWithIndex.foreach(n => {

      val value: Tensor = results(n._2, ---)

      val metric = n._1

      println(metric+": "+value.summarize(maxEntries = value.size.toInt, flattened = true))
      println()
    })
  }

  def generatePlots(): Unit = {}


  /**
    * Has the actual computational logic of producing
    * the metrics which are to be calculated.
    *
    * Implement this method in sub-classes.
    * */
  protected def run(): Tensor

  def to_json: String = {

    val metrics = tfi.unstack(run(), number = names.length, axis = 0)

    val results: Map[String, Any] =
      names.zip(metrics.map(_.entriesIterator.toIndexedSeq)).toMap ++
        Map("shape" -> metrics.head.shape.entriesIterator.toIndexedSeq) ++
        Map("quantity" -> name)

    write_json(results)
  }


}
