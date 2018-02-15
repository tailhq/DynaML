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

import io.github.mandar2812.dynaml.tensorflow.dtf
import org.platanios.tensorflow.api.{Tensor, ::}


/**
  * Top level class for metrics computed on Tensorflow objects.
  *
  * @param preds Predictions
  *
  * @param targets The actual output values.
  * */
abstract class MetricsTF(names: Seq[String], preds: Tensor, targets: Tensor) {

  protected val scoresAndLabels: (Tensor, Tensor) = (preds, targets)

  protected var name = "Target"

  lazy val results: Tensor = run()

  def _target_quantity = name

  def target_quantity_(n: String): Unit = {
    name = n
  }

  def print(): Unit = {
    println("Model Performance: "+name)
    println("============================")
    println()

    names.zipWithIndex.foreach(n => {

      val value: Tensor = results(n._2, ::)

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


}

/**
  * Implements a common use for Regression Task Evaluators.
  * */
class RegressionMetricsTF(preds: Tensor, targets: Tensor)
  extends MetricsTF(Seq("RMSE", "MAE", "Corr"), preds, targets) {


  private lazy val error = targets.subtract(preds)

  private lazy val rmse = error.square.mean(axes = 0).sqrt

  private lazy val mae = error.abs.mean(axes = 0)

  private lazy val corr = {

    val mean_preds = preds.mean(axes = 0, keepDims = true)

    val mean_targets = targets.mean(axes = 0, keepDims = true)

    val preds_c = preds.subtract(mean_preds)

    val targets_c = targets.subtract(mean_targets)

    preds_c.multiply(targets_c).mean(axes = 0).divide(preds_c.square.mean().sqrt).divide(targets_c.square.mean().sqrt)
  }

  override protected def run(): Tensor = dtf.stack(Seq(rmse, mae, corr))
}