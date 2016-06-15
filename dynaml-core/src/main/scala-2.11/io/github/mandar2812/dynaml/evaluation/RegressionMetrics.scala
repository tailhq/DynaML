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
import io.github.mandar2812.dynaml.utils
import org.apache.log4j.{Priority, Logger}

import com.quantifind.charts.Highcharts._

/**
 * Class implementing the calculation
 * of regression performance evaluation
 * metrics
 *
 * */

class RegressionMetrics(
    override protected val scoresAndLabels: List[(Double, Double)],
    val len: Int)
  extends Metrics[Double] {
  private val logger = Logger.getLogger(this.getClass)

  val length: Int = len

  val rmse: Double = math.sqrt(scoresAndLabels.map((p) =>
    math.pow(p._1 - p._2, 2)/length).sum)

  val mae: Double = scoresAndLabels.map((p) =>
    math.abs(p._1 - p._2)/length).sum

  val rmsle: Double = math.sqrt(scoresAndLabels.map((p) =>
    math.pow(math.log(1 + math.abs(p._1)) - math.log(math.abs(p._2) + 1),
      2)/length).sum)

  val Rsq: Double = RegressionMetrics.computeRsq(scoresAndLabels, length)

  val corr: Double = RegressionMetrics.computeCorr(scoresAndLabels, length)

  val predictionEfficiency = scoresAndLabels.map((p) =>
    math.pow(p._1 - p._2, 2)/length).sum

  val modelYield = RegressionMetrics.computeYield(scoresAndLabels, length)

  val sigma: Double =
    math.sqrt(utils.getStats(this.residuals().map(i => DenseVector(i._1)))._2(0)/(length - 1.0))

  def residuals() = this.scoresAndLabels.map((s) => (s._2 - s._1, s._1))

  def scores_and_labels() = this.scoresAndLabels

  override def print(): Unit = {
    logger.info("Regression Model Performance: "+name)
    logger.info("============================")
    logger.info("MAE: " + mae)
    logger.info("RMSE: " + rmse)
    logger.info("RMSLE: " + rmsle)
    logger.info("R^2: " + Rsq)
    logger.info("Corr. Coefficient: " + corr)
    logger.info("Model Yield: "+modelYield)
    logger.info("Std Dev of Residuals: " + sigma)
  }

  override def kpi() = DenseVector(mae, rmse, Rsq, corr)

  override def generatePlots(): Unit = {
    logger.info("Generating Plot of Residuals")
    generateResidualPlot()
    generateFitPlot()
  }

  def generateFitPlot(): Unit = {
    logger.info("Generating plot of goodness of fit")
    regression(scoresAndLabels)
    title("Goodness of fit: "+name)
    xAxis("Predicted "+name)
    yAxis("Actual "+name)
  }

  def generateResidualPlot(): Unit = {
    val roccurve = this.residuals()
    logger.info("Generating plot of residuals vs labels")
    scatter(roccurve.map(i => (i._2, i._1)))
    title("Scatter Plot of Residuals: "+name)
    xAxis("Predicted "+name)
    yAxis("Residual")
  }

}

object RegressionMetrics {
  def computeRsq(scoresAndLabels: Iterable[(Double, Double)], size: Int): Double = {

    val mean: Double = scoresAndLabels.map{coup => coup._2}.sum/size
    var SSres = 0.0
    var SStot = 0.0
    scoresAndLabels.foreach((couple) => {
      SSres += math.pow(couple._2 - couple._1, 2)
      SStot += math.pow(couple._2 - mean, 2)
    })
    1 - (SSres/SStot)
  }

  def computeCorr(scoresAndLabels: Iterable[(Double, Double)], size: Int): Double = {

    val meanLabel: Double = scoresAndLabels.map{coup => coup._2}.sum/size
    val meanScore = scoresAndLabels.map{coup => coup._1}.sum/size
    var SSLabel = 0.0
    var SSPred = 0.0
    var SSLabelPred = 0.0
    scoresAndLabels.foreach((couple) => {
      SSLabel += math.pow(couple._2 - meanLabel, 2)
      SSPred += math.pow(couple._1 - meanScore, 2)
      SSLabelPred += (couple._1 - meanScore)*(couple._2 - meanLabel)
    })

    SSLabelPred/(math.sqrt(SSPred)*math.sqrt(SSLabel))
  }

  def computeYield(scoresAndLabels: Iterable[(Double, Double)], size: Int): Double =
    (scoresAndLabels.map(_._1).max - scoresAndLabels.map(_._1).min)/
      (scoresAndLabels.map(_._2).max - scoresAndLabels.map(_._2).min)

}
