package org.kuleuven.esat.evaluation

import breeze.linalg.DenseVector
import org.apache.log4j.{Priority, Logger}

import scalax.chart.module.ChartFactories.XYLineChart

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

  private val bias: Double = scoresAndLabels.map{v => v._1}.sum/length
  private val thresholds = (-1000 to 1000).map((x) => (x.toDouble + bias)/100)


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
      var tp: Double = 0.0
      var fp: Double = 0.0
      var count: Double = 0.0
      this.scoresAndLabels.foreach((couple) => {
        count += 1.0
        if(math.signum(couple._1 - th) == couple._2) {
          tp += 1.0
        } else {
          fp += 1.0
        }
      })
      (th, (tp/count, fp/count))
    })

  /**
   * Generate the PR, ROC and F1 measure
   * plots using Scala-Chart.
   * */
  override def generatePlots(): Unit = {
    val roccurve = this.roc()
    val prcurve = this.pr()
    val fm = this.fMeasureByThreshold()

    logger.log(Priority.INFO, "Generating ROC Plot")
    val chart1 = XYLineChart(roccurve,
      title = "Receiver Operating Characteristic", legend = true)
    chart1.show()

    logger.log(Priority.INFO, "Generating PR Plot")
    val chart2 = XYLineChart(prcurve,
      title = "Precision Recall Curve", legend = true)
    chart2.show()

    logger.log(Priority.INFO, "Generating F1 measure Plot")
    val chart3 = XYLineChart(fm,
      title = "F1 measure by threshold beta = 1", legend = true)
    chart3.show()
  }

  override def print(): Unit = {
    logger.log(Priority.INFO, "Classification Model Performance")
    logger.log(Priority.INFO, "============================")
    logger.log(Priority.INFO, "Area under PR: " + areaUnderPR())
    logger.log(Priority.INFO, "Area under ROC: " + areaUnderROC())

  }

  override def kpi() = DenseVector(areaUnderPR(),
    areaUnderROC(),
    fMeasureByThreshold().map((c) => c._2).max)
}
