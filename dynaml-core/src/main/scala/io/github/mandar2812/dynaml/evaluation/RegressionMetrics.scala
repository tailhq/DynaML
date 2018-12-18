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
import breeze.numerics.{abs, log, sqrt}
import io.github.mandar2812.dynaml.utils
import org.apache.log4j.{Logger, Priority}
import com.quantifind.charts.Highcharts._
import io.github.mandar2812.dynaml.algebra.square

/**
 * Class implementing the calculation
 * of regression performance evaluation
 * metrics
 *
 * */

class RegressionMetrics(
    override protected val scoresAndLabels: List[(Double, Double)],
    val len: Int) extends Metrics[Double] {

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

  val sp_corr: Double = RegressionMetrics.computeSpearmanCorr(scoresAndLabels, length)

  val predictionEfficiency = scoresAndLabels.map((p) =>
    math.pow(p._1 - p._2, 2)/length).sum

  val modelYield = RegressionMetrics.computeYield(scoresAndLabels, length)

  val sigma: Double =
    math.sqrt(utils.getStats(this.residuals().map(i => DenseVector(i._1)))._2(0)/(length - 1.0))

  def residuals() = this.scoresAndLabels.map((s) => (s._2 - s._1, s._1))

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

    scala.Predef.print("R^2 = ")
    pprint.pprintln(Rsq)

    scala.Predef.print("Corr. Coefficient = ")
    pprint.pprintln(corr)

    scala.Predef.print("Spearman Corr. Coefficient = ")
    pprint.pprintln(sp_corr)

    scala.Predef.print("Model Yield = ")
    pprint.pprintln(modelYield)

    scala.Predef.print("Std Dev of Residuals = ")
    pprint.pprintln(sigma)
  }

  override def kpi() = DenseVector(mae, rmse, Rsq, corr, sp_corr)

  override def generatePlots(): Unit = {
    println("Generating Plot of Residuals")
    generateResidualPlot()
    generateFitPlot()
  }

  def generateFitPlot(): Unit = {
    println("Generating plot of goodness of fit")
    regression(scoresAndLabels)
    title("Goodness of fit: "+name)
    xAxis("Predicted "+name)
    yAxis("Actual "+name)
  }

  def generateResidualPlot(): Unit = {
    val roccurve = this.residuals()
    println("Generating plot of residuals vs labels")
    scatter(roccurve.map(i => (i._2, i._1)))
    title("Scatter Plot of Residuals: "+name)
    xAxis("Predicted "+name)
    yAxis("Residual")
  }

  def ++(otherMetrics: RegressionMetrics): RegressionMetrics = {
    new RegressionMetrics(
      this.scoresAndLabels ++ otherMetrics.scoresAndLabels,
      this.length + otherMetrics.length).setName(this.name)
  }
}

object RegressionMetrics {
  def computeRsq(scoresAndLabels: Iterable[(Double, Double)], size: Int): Double = {

    val mean: Double = scoresAndLabels.map{coup => coup._2}.sum/size
    var SSres = 0.0
    var SStot = 0.0
    scoresAndLabels.foreach(couple => {
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
    scoresAndLabels.foreach(couple => {
      SSLabel += math.pow(couple._2 - meanLabel, 2)
      SSPred += math.pow(couple._1 - meanScore, 2)
      SSLabelPred += (couple._1 - meanScore)*(couple._2 - meanLabel)
    })

    SSLabelPred/(math.sqrt(SSPred)*math.sqrt(SSLabel))
  }

  def computeSpearmanCorr(scoresAndLabels: Iterable[(Double, Double)], size: Int): Double = {

    val (predictions, targets) = scoresAndLabels.toSeq.unzip

    val addOneToIndex = (p: (Double, Int)) => (p._1, p._2 + 1)

    def sort_with_ranks(seq: Seq[Double]) =
      seq.sorted
        .zipWithIndex
        .map(addOneToIndex)
        .groupBy(_._1)
        .map(coll => (coll._1, coll._2.map(_._2)))
        .mapValues(values => values.sum.toDouble/values.length)

    val preds_rank_map   = sort_with_ranks(predictions)
    val targets_rank_map = sort_with_ranks(targets)

    val ranks_preds      = predictions.map(preds_rank_map(_))
    val ranks_targets    = targets.map(targets_rank_map(_))

    computeCorr(ranks_preds.zip(ranks_targets), size)
  }

  def computeYield(scoresAndLabels: Iterable[(Double, Double)], size: Int): Double =
    (scoresAndLabels.map(_._1).max - scoresAndLabels.map(_._1).min)/
      (scoresAndLabels.map(_._2).max - scoresAndLabels.map(_._2).min)

}

class MultiRegressionMetrics(
  override protected val scoresAndLabels: List[(DenseVector[Double], DenseVector[Double])],
  val len: Int) extends Metrics[DenseVector[Double]] {


  val num_outputs: Int = scoresAndLabels.head._2.length

  val onesVec = DenseVector.ones[Double](num_outputs)

  val length: DenseVector[Double] = DenseVector.fill(num_outputs)(len)

  val rmse: DenseVector[Double] = sqrt(scoresAndLabels.map((p) =>
    square(p._1-p._2)).reduce((a: DenseVector[Double],b:DenseVector[Double]) => a+b)/length)

  val mae: DenseVector[Double] = scoresAndLabels.map((p) =>
    abs(p._1 - p._2)).reduce((a: DenseVector[Double],b:DenseVector[Double]) => a+b)/length

  val rmsle: DenseVector[Double] = sqrt(scoresAndLabels.map((p) =>
    square(log(onesVec + abs(p._1)) - log(abs(p._2) + onesVec)))
    .reduce((a: DenseVector[Double],b:DenseVector[Double]) => a+b)/length)

  val Rsq: DenseVector[Double] = MultiRegressionMetrics.computeRsq(scoresAndLabels, length)

  val corr: DenseVector[Double] = MultiRegressionMetrics.computeCorr(scoresAndLabels, length)

  val predictionEfficiency = scoresAndLabels.map((p) =>
    square(p._1 - p._2)).reduce((a: DenseVector[Double],b:DenseVector[Double]) => a+b)/length

  val modelYield: DenseVector[Double] = MultiRegressionMetrics.computeYield(scoresAndLabels, length)

  val sigma: DenseVector[Double] =
    sqrt(utils.getStats(this.residuals().map(_._1))._2/(length - 1.0))

  def residuals() = this.scoresAndLabels.map((s) => (s._2 - s._1, s._1))

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

    scala.Predef.print("R^2 = ")
    pprint.pprintln(Rsq)

    scala.Predef.print("Corr. Coefficient = ")
    pprint.pprintln(corr)

    scala.Predef.print("Model Yield = ")
    pprint.pprintln(modelYield)

    scala.Predef.print("Std Dev of Residuals = ")
    pprint.pprintln(sigma)
  }

  override def kpi() = DenseVector(mae, rmse, Rsq, corr)

  override def generatePlots(): Unit = {
    println("Generating Plot of Fit for each target")
    (0 until num_outputs).foreach(output => {
      regression(scoresAndLabels.map(couple => (couple._1(output), couple._2(output))))
    })
    title("Goodness of fit: "+name)
    xAxis("Predicted "+name)
    yAxis("Actual "+name)
  }

  def ++(otherMetrics: MultiRegressionMetrics): MultiRegressionMetrics = {
    new MultiRegressionMetrics(
      this.scoresAndLabels ++ otherMetrics.scoresAndLabels,
      this.len + otherMetrics.len).setName(this.name)
  }

}

object MultiRegressionMetrics {
  def computeRsq(scoresAndLabels: Iterable[(DenseVector[Double], DenseVector[Double])],
                 size: DenseVector[Double]): DenseVector[Double] = {

    val num_outputs = scoresAndLabels.head._2.length
    val mean: DenseVector[Double] =
      scoresAndLabels.map{_._2}.reduce((a: DenseVector[Double],b:DenseVector[Double]) => a+b):/size

    var SSres = DenseVector.zeros[Double](num_outputs)
    var SStot = DenseVector.zeros[Double](num_outputs)

    scoresAndLabels.foreach((couple) => {
      SSres :+= square(couple._2 - couple._1)
      SStot :+= square(couple._2 - mean)
    })

    DenseVector.ones[Double](num_outputs) - (SSres:/SStot)
  }

  def computeCorr(scoresAndLabels: Iterable[(DenseVector[Double], DenseVector[Double])],
                  size: DenseVector[Double]): DenseVector[Double] = {

    val num_outputs = scoresAndLabels.head._2.length

    val meanLabel: DenseVector[Double] = scoresAndLabels.map{_._2}
      .reduce((a: DenseVector[Double],b:DenseVector[Double]) => a+b):/size

    val meanScore = scoresAndLabels.map{_._1}
      .reduce((a: DenseVector[Double],b:DenseVector[Double]) => a+b):/size

    var SSLabel = DenseVector.zeros[Double](num_outputs)
    var SSPred = DenseVector.zeros[Double](num_outputs)
    var SSLabelPred = DenseVector.zeros[Double](num_outputs)

    scoresAndLabels.foreach((couple) => {
      SSLabel :+= square(couple._2 - meanLabel)
      SSPred :+= square(couple._1 - meanScore)
      SSLabelPred :+= (couple._1 - meanScore) :* (couple._2 - meanLabel)
    })

    SSLabelPred:/(sqrt(SSPred):*sqrt(SSLabel))
  }

  def computeYield(scoresAndLabels: Iterable[(DenseVector[Double], DenseVector[Double])],
                   size: DenseVector[Double]): DenseVector[Double] = {
    val num_outputs = scoresAndLabels.head._2.length
    DenseVector.tabulate[Double](num_outputs)(dimension => {
      //for each dimension, calculate the model yield
      RegressionMetrics.computeYield(
        scoresAndLabels.map(c => (c._1(dimension), c._2(dimension))),
        size(0).toInt)
    })
  }
}

