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
    val len: Int, logisticFlag: Boolean = false)
  extends Metrics[Double]{

  private val logger = Logger.getLogger(this.getClass)
  val length = len

  /**
   * A list of threshold values from
   * min(score) to max(score) in 100 steps. These
   * will be used to measure the variation
   * in precision, recall False Positive
   * and False Negative values.
   */
  private val thresholds = List.tabulate(100)(i => {
    scoresAndLabels.map(_._1).min +
      i.toDouble*((scoresAndLabels.map(_._1).max.toInt -
        scoresAndLabels.map(_._1).min.toInt + 1)/100.0)})

  val (positives, negatives) =  logisticFlag match {
    case true =>
      (scoresAndLabels.filter(_._2 == 1.0),
      scoresAndLabels.filter(_._2 == 0.0))
    case false =>
      (scoresAndLabels.filter(_._2 == 1.0),
      scoresAndLabels.filter(_._2 == -1.0))
  }

  def scores_and_labels = this.scoresAndLabels

  private def areaUnderCurve(points: List[(Double, Double)]): Double =
    points.sliding(2).map(l => (l(1)._1 - l.head._1) * (l(1)._2 + l.head._2)/2).sum

  /**
   * Calculate the area under the Precision-Recall
   * curve.
   */
  def areaUnderPR(): Double = areaUnderCurve(this.pr())

  /**
   * Calculate the area under the Receiver
   * Operating Characteristic curve.
   */
  def areaUnderROC(): Double = areaUnderCurve(this.roc())

  /**
   * Calculate the F1 metric by threshold, for a
   * beta value of 1.0
   */
  def fMeasureByThreshold(): List[(Double, Double)] = fMeasureByThreshold(1.0)

  /**
   * Calculate the F1 metric by threshold, for an
   * arbitrary beta value
   * */
  def fMeasureByThreshold(beta: Double): List[(Double, Double)] =
    tpfpByThreshold().map((couple) => {
      val tp = couple._2._1
      val fp = couple._2._2
      val betasq = math.pow(beta, 2.0)
      (couple._1, (1 + betasq)*tp/((1 + betasq)*tp + betasq*(1-tp) + fp))
    })

  /**
   * Return the Precision-Recall curve, as a [[List]]
   * of [[Tuple2]] (Recall, Precision).
   */
  def pr(): List[(Double, Double)] =
    recallByThreshold().zip(precisionByThreshold()).map((couple) =>
      (couple._1._2, couple._2._2)).sorted

  /**
   * Return the Recall-Threshold curve, as a [[List]]
   * of [[Tuple2]] (Threshold, Recall).
   */
  def recallByThreshold(): List[(Double, Double)] =
    tpfpByThreshold().map((point) => (point._1, point._2._1))

  /**
   * Return the Precision-Threshold curve, as a [[List]]
   * of [[Tuple2]] (Threshold, Precision).
   */
  def precisionByThreshold(): List[(Double, Double)] =
    tpfpByThreshold().map((point) =>
      (point._1, point._2._1/(point._2._1 + point._2._2)))

  /**
   * Return the Receiver Operating Characteristic
   * curve, as a [[List]] of [[Tuple2]]
   * (False Positive Rate, True Positive Rate).
   */
  def roc(): List[(Double, Double)] =
    tpfpByThreshold().map((point) => (point._2._2, point._2._1)).sorted

  /**
   * Return the True Positive and False Positive Rate
   * with respect to the threshold, as a [[List]]
   * of [[Tuple2]] (Threshold, (True Positive rate, False Positive Rate)).
   */
  def tpfpByThreshold(): List[(Double, (Double, Double))]  =
    this.thresholds.map((th) => {
      val true_positive = if(positives.nonEmpty) {
        positives.count(p =>
          math.signum(p._1 - th) == 1.0)
          .toDouble/positives.length
      } else {0.0}

      val false_positive = if(negatives.nonEmpty) {
        negatives.count(p =>
          math.signum(p._1 - th) == 1.0)
          .toDouble/negatives.length
      } else {0.0}

      (th, (true_positive, false_positive))
    })

  /**
    * Return the True Positive and False Positive Rate
    * with respect to the threshold, as a [[List]]
    * of [[Tuple2]]
    * (Threshold, (True Positive rate, True Negative Rate, False Positive Rate, False Negative Rate)).
    */
  def tptn_fpfnByThreshold: List[(Double, (Double, Double, Double, Double))]  =
  this.thresholds.map((th) => {
    val (true_positive, false_negative) = if(positives.nonEmpty) {
      val t = positives.partition(p => math.signum(p._1 - th) == 1.0)

      (t._1.length.toDouble/positives.length, t._2.length.toDouble/positives.length)
    } else {(0.0, 0.0)}

    val (false_positive, true_negative) = if(negatives.nonEmpty) {
      val f = negatives.partition(p => math.signum(p._1 - th) == 1.0)
      (f._1.length.toDouble/positives.length, f._2.length.toDouble/positives.length)
    } else {(0.0, 0.0)}

    (th, (true_positive, true_negative, false_positive, false_negative))
  })

  /**
    * Returns the Matthew's correlation coefficient
    * for every thresholding value.
    *
    */
  def matthewsCCByThreshold: List[(Double, Double)] = tptn_fpfnByThreshold.map(t => {
    val (tp, tn, fp, fn) = t._2
    (t._1, (tp*tn-(fp*fn))/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
  })

  def accuracyByThreshold(): List[(Double, Double)] = tpfpByThreshold().map((t) => (t._1,
    (t._2._1*positives.length + (1.0-t._2._2)*negatives.length)/scoresAndLabels.length))

  /**
   * Generate the PR, ROC and F1 measure
   * plots using Scala-Chart.
   */
  override def generatePlots(): Unit = {
    val roccurve = this.roc()
    val prcurve = this.pr()
    val fm = this.fMeasureByThreshold()
    val mtt = matthewsCCByThreshold

    println("Generating Matthew's correlation coefficient plot by thresholding value")
    spline(mtt.map(_._1), mtt.map(_._2))
    title("MCC vs Threshold Cutoff: "+name)
    xAxis("Threshold")
    yAxis("F Measure")

    println("Generating F1-measure plot by thresholding value")
    spline(fm.map(_._1), fm.map(_._2))
    title("F Measure vs Threshold Cutoff: "+name)
    xAxis("Threshold")
    yAxis("F Measure")

    println("Generating ROC Plot")
    areaspline(roccurve.map(_._1), roccurve.map(_._2))
    title("Receiver Operating Characteristic: "+name+
      ", Area under curve: "+areaUnderCurve(roccurve))
    xAxis("False Positives")
    yAxis("True Positives")
  }

  override def print(): Unit = {
    println("Classification Model Performance: "+name)
    println("============================")
    scala.Predef.print("Accuracy = ")
    pprint.pprintln(accuracyByThreshold().map((c) => c._2).max)


    scala.Predef.print("Area under ROC = ")
    pprint.pprintln(areaUnderROC())

    scala.Predef.print("Maximum F Measure = ")
    pprint.pprintln(fMeasureByThreshold().map(_._2).max)

    scala.Predef.print("Maximum Matthew's Correlation Coefficient = ")
    pprint.pprintln(matthewsCCByThreshold.map(_._2).max)
  }

  override def kpi() = DenseVector(accuracyByThreshold().map((c) => c._2).max,
    fMeasureByThreshold().map((c) => c._2).max,
    areaUnderROC())

  def ++(otherMetrics: BinaryClassificationMetrics): BinaryClassificationMetrics = {
    new BinaryClassificationMetrics(
      this.scoresAndLabels ++ otherMetrics.scoresAndLabels,
      this.length + otherMetrics.length).setName(this.name)
  }
}
