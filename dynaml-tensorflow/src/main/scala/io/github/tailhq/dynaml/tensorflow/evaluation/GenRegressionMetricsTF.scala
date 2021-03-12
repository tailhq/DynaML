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
package io.github.tailhq.dynaml.tensorflow.evaluation

import io.github.tailhq.dynaml.graphics.charts.Highcharts.{regression, title, xAxis, yAxis}
import io.github.tailhq.dynaml.tensorflow.dtf
import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, IsReal, TF}

/**
  * Generalisation of [[RegressionMetricsTF]] to more complex output structures, like
  * matrix outputs/labels.
  *
  * @author tailhq date: 9/03/2018
  * */
class GenRegressionMetricsTF[D: TF: IsNotQuantized: IsReal](preds: Tensor[D], targets: Tensor[D]) extends
  RegressionMetricsTF(preds, targets) {
  private val num_outputs: Int =
    if (preds.shape.toTensor.size == 1) 1
    else preds.shape.toTensor(0 :: -1).prod[Int]().scalar.toInt

  private lazy val (_ , rmse , mae, corr) = GenRegressionMetricsTF.calculate(preds, targets)

  private lazy val modelyield =
    (preds.max(axes = 0) - preds.min(axes = 0)).divide(targets.max(axes = 0) - targets.min(axes = 0))

  override protected def run(): Tensor[D] = dtf.stack(Seq(rmse, mae, corr, modelyield), axis = -1)

  override def generatePlots(): Unit = {
    println("Generating Plot of Fit for each target")

    if(num_outputs == 1) {
      val (pr, tar) = (
        scoresAndLabels._1.entriesIterator.map(_.asInstanceOf[Float]),
        scoresAndLabels._2.entriesIterator.map(_.asInstanceOf[Float]))

      regression(pr.zip(tar).toSeq)

      title("Goodness of fit: "+name)
      xAxis("Predicted "+name)
      yAxis("Actual "+name)

    } else {
      (0 until num_outputs).foreach(output => {
        val (pr, tar) = (
          scoresAndLabels._1(::, output).entriesIterator.map(_.asInstanceOf[Float]),
          scoresAndLabels._2(::, output).entriesIterator.map(_.asInstanceOf[Float]))

        regression(pr.zip(tar).toSeq)
      })
    }
  }
}

object GenRegressionMetricsTF {

  protected def calculate[D : TF: IsNotQuantized: IsReal](
    preds: Tensor[D], targets: Tensor[D]): (Tensor[D], Tensor[D], Tensor[D], Tensor[D]) = {
    val error = targets.subtract(preds)

    println("Shape of error tensor: "+error.shape.toString()+"\n")

    val rmse = error.square.mean(axes = 0).sqrt

    val mae = error.abs.mean(axes = 0)

    val corr = {

      val mean_preds = preds.mean(axes = 0)

      val mean_targets = targets.mean(axes = 0)

      val preds_c = preds.subtract(mean_preds)

      val targets_c = targets.subtract(mean_targets)

      val (sigma_t, sigma_p) = (targets_c.square.mean(axes = 0).sqrt, preds_c.square.mean(axes = 0).sqrt)

      preds_c.multiply(targets_c).mean(axes = 0).divide(sigma_t.multiply(sigma_p))
    }

    (error, rmse, mae, corr)
  }
}