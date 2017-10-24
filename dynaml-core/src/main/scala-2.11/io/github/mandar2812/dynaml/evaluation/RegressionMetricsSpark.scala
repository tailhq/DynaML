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
import com.quantifind.charts.Highcharts._
import org.apache.log4j.{Priority, Logger}
import org.apache.spark.Accumulator
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scalax.chart.module.ChartFactories.{XYBarChart, XYLineChart, XYAreaChart}

/**
 * Class implementing the calculation
 * of regression performance evaluation
 * metrics
 *
 * */

class RegressionMetricsSpark(protected val scores: RDD[(Double, Double)],
                             val len: Long)
  extends Metrics[Double] {

  override protected val scoresAndLabels = List()
  private val logger = Logger.getLogger(this.getClass)
  val length = len

  val (mae, rmse, rsq, rmsle):(Double, Double, Double, Double) = 
    RegressionMetricsSpark.computeKPIs(scores, length)
  
  def residuals() = this.scores.map((s) => (s._1 - s._2, s._2))

  def scores_and_labels() = this.scoresAndLabels

  override def print(): Unit = {
    println("Regression Model Performance: "+name)
    println("============================")

    scala.Predef.print("MAE = ")
    pprint.pprintln(mae)

    scala.Predef.print("RMSE = ")
    pprint.pprintln(rmse)

    scala.Predef.print("RMSLE = ")
    pprint.pprintln(rmsle)

/*
    scala.Predef.print("R^2 = ")
    pprint.pprintln(Rsq)

    scala.Predef.print("Corr. Coefficient = ")
    pprint.pprintln(corr)

    scala.Predef.print("Model Yield = ")
    pprint.pprintln(modelYield)

    scala.Predef.print("Std Dev of Residuals = ")
    pprint.pprintln(sigma)
*/
  }

  override def kpi() = DenseVector(mae, rmse, rsq)

  override def generatePlots(): Unit = {
    implicit val theme = org.jfree.chart.StandardChartTheme.createDarknessTheme
    val residuals = this.residuals().map(_._1).collect().toList

    println("Generating Plot of Residuals")
    /*val chart1 = XYBarChart(roccurve,
      title = "Residuals", legend = true)

    chart1.show()*/
    histogram(residuals, numBins = 20)
    title("Histogram of Regression Residuals")
  }

}

object RegressionMetricsSpark {

  def computeKPIs(scoresAndLabels: RDD[(Double, Double)], size: Long)
  : (Double, Double, Double, Double) = {
    val mean: Accumulator[Double] = scoresAndLabels.context.accumulator(0.0, "mean")

    val err:DenseVector[Double] = scoresAndLabels.map((sc) => {
      val diff = sc._1 - sc._2
      mean += sc._2
      val difflog = math.pow(math.log(1 + math.abs(sc._1)) - math.log(math.abs(sc._2) + 1),
        2)
      DenseVector(math.abs(diff), math.pow(diff, 2.0), difflog)
    }).reduce((a,b) => a+b)

    val SS_res = err(1)

    val mu: Broadcast[Double] = scoresAndLabels.context.broadcast(mean.value/size.toDouble)

    val SS_tot = scoresAndLabels.map((sc) => math.pow(sc._2 - mu.value, 2.0)).sum()

    val rmse = math.sqrt(SS_res/size.toDouble)
    val mae = err(0)/size.toDouble
    val rsq = if(1/SS_tot != Double.NaN) 1 - (SS_res/SS_tot) else 0.0
    val rmsle = err(2)/size.toDouble
    (mae, rmse, rsq, rmsle)
  } 
  
}
