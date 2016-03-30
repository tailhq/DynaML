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

import breeze.linalg.DenseVector
import org.apache.spark.rdd.RDD

/**
 * Abstract trait for metrics
 */
trait Metrics[P] {
  protected val scoresAndLabels: List[(P, P)]
  protected var name = "Value"
  def print(): Unit
  def generatePlots(): Unit = {}
  def kpi(): DenseVector[P]
  def setName(n: String): this.type = {
    name = n
    this
  }
}

object Metrics{
  def apply(task: String)
           (scoresAndLabels: List[(Double, Double)], length: Int)
  : Metrics[Double] = task match {
    case "regression" => new RegressionMetrics(scoresAndLabels, length)
    case "classification" => new BinaryClassificationMetrics(scoresAndLabels, length)
  }
}

object MetricsSpark {
  def apply(task: String)
           (scoresAndLabels: RDD[(Double, Double)],
            length: Long,
            minmax: (Double, Double))
  : Metrics[Double] = task match {
    case "regression" => new RegressionMetricsSpark(scoresAndLabels, length)
    case "classification" => new BinaryClassificationMetricsSpark(scoresAndLabels, length, minmax)
  }
}
