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
import org.apache.log4j.{Priority, Logger}

import com.quantifind.charts.Highcharts._

/**
 * Class implementing the calculation
 * of different binary classification
 * performance metrics
 *
 * */

class BinaryClassificationMetrics(
    override protected val scoresAndLabels: List[(Double, Double)],
    val len: Int)
  extends Metrics[Double]{

  private val logger = Logger.getLogger(this.getClass)
  val length = len

  /**
   * A list of threshold values from
   * -1.0 to 1.0 in 100 steps. These
   * will be used to measure the variation
   * in precision, recall False Positive
   * and False Negative values.
   * */

  private val thresholds = List.tabulate(100)(i => {
    scoresAndLabels.map(_._1).min +
      i.toDouble*((scoresAndLabels.map(_._1).max.toInt -
        scoresAndLabels.map(_._1).min.toInt + 1)/100.0)})

  val positives = this.scoresAndLabels.filter(_._2 == 1.0)
  val negatives = this.scoresAndLabels.filter(_._2 == -1.0)

  def scores_and_labels = this.scoresAndLabels

  private def areaUnderCurve(points: List[(Double, Double)]): Double =
    points.sliding(2).map(l => (l(1)._1 - l(0)._1) * (l(1)._2 + l(0)._2)/2).sum

  /**
   * Calculate the area under the Precision-Recall
   * curve.
   * */
  def areaUnderPR(): Double = areaUnderCurve(this.pr())

  /**
   * Calculate the area under the Receiver
   * Operating Characteristic curve.
   * */
  def areaUnderROC(): Double = areaUnderCurve(this.roc())

  /**
   * Calculate the F1 metric by threshold, for a
   * beta value of 1.0
   * */
  def fMeasureByThreshold(): List[(Double, Double)] = fMeasureByThreshold(1.0)

  /**
   * Calculate the F1 metric by threshold, for an
   * arbitrary beta value
   * */
  def fMeasureByThreshold(beta: Double): List[(Double, Double)] = tpfpByThreshold().map((couple) => {
    val tp = couple._2._1
    val fp = couple._2._2
    val betasq = math.pow(beta, 2.0)
    (couple._1, (1 + betasq)*tp/((1 + betasq)*tp + betasq*(1-tp) + fp))
  })

  /**
   * Return the Precision-Recall curve, as a [[List]]
   * of [[Tuple2]] (Recall, Precision).
   * */
  def pr(): List[(Double, Double)] = recallByThreshold().zip(precisionByThreshold()).map((couple) =>
    (couple._1._2, couple._2._2)).sorted

  /**
   * Return the Recall-Threshold curve, as a [[List]]
   * of [[Tuple2]] (Threshold, Recall).
   * */
  def recallByThreshold(): List[(Double, Double)]  = tpfpByThreshold().map((point) => (point._1, point._2._1))

  /**
   * Return the Precision-Threshold curve, as a [[List]]
   * of [[Tuple2]] (Threshold, Precision).
   * */
  def precisionByThreshold(): List[(Double, Double)]  = tpfpByThreshold().map((point) =>
    (point._1, point._2._1/(point._2._1 + point._2._2)))

  /**
   * Return the Receiver Operating Characteristic
   * curve, as a [[List]] of [[Tuple2]]
   * (False Positive Rate, True Positive Rate).
   * */
  def roc(): List[(Double, Double)] =
    tpfpByThreshold().map((point) => (point._2._2, point._2._1)).sorted

  /**
   * Return the True Positive and False Positive Rate
   * with respect to the threshold, as a [[List]]
   * of [[Tuple2]] (Threshold, (True Positive rate, False Positive Rate)).
   * */
  def tpfpByThreshold(): List[(Double, (Double, Double))]  =
    this.thresholds.toList.map((th) => {
      val true_positive = if(positives.length > 0) {
        positives.count(p =>
          math.signum(p._1 - th) == 1.0)
          .toDouble/positives.length
      } else {0.0}

      val false_positive = if(negatives.length > 0) {
        negatives.count(p =>
          math.signum(p._1 - th) == 1.0)
          .toDouble/negatives.length
      } else {0.0}

      (th, (true_positive, false_positive))
    })

  def accuracyByThreshold(): List[(Double, Double)] = tpfpByThreshold().map((t) => (t._1,
    (t._2._1*positives.length + (1.0-t._2._2)*negatives.length)/scoresAndLabels.length))

  /**
   * Generate the PR, ROC and F1 measure
   * plots using Scala-Chart.
   * */
  override def generatePlots(): Unit = {
    val roccurve = this.roc()
    val prcurve = this.pr()
    val fm = this.fMeasureByThreshold()
    //implicit val theme = org.jfree.chart.StandardChartTheme.createDarknessTheme
    logger.log(Priority.INFO, "Generating ROC Plot")
    /*val chart1 = XYAreaChart(roccurve,
      title = "Receiver Operating Characteristic", legend = true)

    chart1.show()

    logger.log(Priority.INFO, "Generating PR Plot")
    val chart2 = XYAreaChart(prcurve,
      title = "Precision Recall Curve", legend = true)
    chart2.show()

    logger.log(Priority.INFO, "Generating F1 measure Plot")
    val chart3 = XYLineChart(fm,
      title = "F1 measure by threshold beta = 1", legend = true)
    chart3.show()*/
    areaspline(roccurve.map(_._1), roccurve.map(_._2))
    title("Receiver Operating Characteristic")
    xAxis("False Positives")
    yAxis("True Positives")

  }

  override def print(): Unit = {
    logger.log(Priority.INFO, "Classification Model Performance")
    logger.log(Priority.INFO, "============================")
    logger.log(Priority.INFO, "Accuracy: " + accuracyByThreshold().map((c) => c._2).max)
    logger.log(Priority.INFO, "Area under ROC: " + areaUnderROC())

  }

  override def kpi() = DenseVector(accuracyByThreshold().map((c) => c._2).max,
    fMeasureByThreshold().map((c) => c._2).max,
    areaUnderROC())
}
