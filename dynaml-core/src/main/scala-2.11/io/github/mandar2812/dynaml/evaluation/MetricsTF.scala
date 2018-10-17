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

import org.platanios.tensorflow.api.types.{DecimalDataType, MathDataType}
import org.platanios.tensorflow.api.{---, ::, Tensor}


/**
  * Top level class for metrics computed on Tensorflow objects.
  *
  * @param preds Predictions
  *
  * @param targets The actual output values.
  * */
abstract class MetricsTF[D <: DecimalDataType](val names: Seq[String], val preds: Tensor[D], val targets: Tensor[D]) {

  protected val scoresAndLabels: (Tensor[D], Tensor[D]) = (preds, targets)

  protected var name = "Target"

  lazy val results: Tensor[D] = run()

  def _target_quantity: String = name

  def target_quantity_(n: String): Unit = {
    name = n
  }

  def print(): Unit = {
    println("\nModel Performance: "+name)
    println("============================")
    println()

    names.zipWithIndex.foreach(n => {

      val value: Tensor[D] = results(n._2, ---)

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
  protected def run(): Tensor[D]


}
