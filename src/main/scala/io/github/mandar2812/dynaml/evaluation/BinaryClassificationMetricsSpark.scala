package io.github.mandar2812.dynaml.evaluation

import breeze.linalg.DenseVector
import org.apache.log4j.Logger
import org.apache.log4j.Logger
import org.apache.log4j.Priority
import org.apache.log4j.{Priority, Logger}
import org.apache.spark.mllib.rdd.RDDFunctions
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD

import scalax.chart.module.ChartFactories.{XYAreaChart, XYLineChart}

/**
 * Class implementing the calculation
 * of different binary classification
 * performance metrics
 *
 * */

class BinaryClassificationMetricsSpark(protected val scores: RDD[(Double, Double)],
                                       val len: Long)
  extends Metrics[Double]{

  override protected val scoresAndLabels = List()

  private val logger = Logger.getLogger(this.getClass)
  val length = len

  /**
   * A list of threshold values from
   * -1.0 to 1.0 in 100 steps. These
   * will be used to measure the variation
   * in precision, recall False Positive
   * and False Negative values.
   * */

  private val (scMin, scMax) = scores.map(s => {(s._1,s._1)}).reduce((couple1, couple2) => {
    (math.min(couple1._1, couple2._1), math.max(couple1._2, couple2._2))
  })

  private val thresholds = List.tabulate(100)(i => {
    scMin +
      i.toDouble*((scMax.toInt -
        scMin.toInt + 1)/100.0)})

  def scores_and_labels = this.scoresAndLabels

  private def areaUnderCurve(points: List[(Double, Double)]): Double =
    points.sliding(2)
      .map(l => (l(1)._1 - l.head._1) * (l(1)._2 + l.head._2)/2).sum


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
    tpfpByThreshold().map((point) => (point._2._2, point._2._1))

  /**
   * Return the True Positive and False Positive Rate
   * with respect to the threshold, as a [[List]]
   * of [[Tuple2]] (Threshold, (True Positive rate, False Positive Rate)).
   * */
  def tpfpByThreshold(): List[(Double, (Double, Double))]  =
  {
    val positives = scores.context.accumulator(0.0, "positives")
    val negatives = scores.context.accumulator(0.0, "negatives")
    val ths = scores.context.broadcast(thresholds.length)
    val thres = scores.context.broadcast(thresholds)
    val (tp, fp) = this.scores.map((sl) =>{

      val (tpv, fpv): (DenseVector[Double], DenseVector[Double]) =
        if(sl._2 == 1.0) {
          positives += 1.0
          (DenseVector.tabulate(ths.value)(i => {
            if(math.signum(sl._1 - thres.value(i)) == sl._2) 1.0 else 0.0
          }), DenseVector.zeros(ths.value))

        } else {
          negatives += 1.0
          (DenseVector.zeros(ths.value), DenseVector.tabulate(ths.value)(i => {
            if(math.signum(sl._1 - thres.value(i)) == 1.0) 1.0 else 0.0
          }))
        }
      (tpv,fpv)
    }).reduce((c1, c2) => {
      (c1._1+c2._1, c1._2+c2._2)
    })
    val pos = positives.value
    val neg = negatives.value
    ths.destroy()
    thres.destroy()
    List.tabulate(thresholds.length){t => {
      (thresholds(t), (tp(t)/pos.toDouble, fp(t)/neg.toDouble))
    }}
  }

  /**
   * Generate the PR, ROC and F1 measure
   * plots using Scala-Chart.
   * */
  override def generatePlots(): Unit = {
    val roccurve = this.roc()
    val prcurve = this.pr()
    val fm = this.fMeasureByThreshold()
    implicit val theme = org.jfree.chart.StandardChartTheme.createDarknessTheme
    logger.log(Priority.INFO, "Generating ROC Plot")
    val chart1 = XYAreaChart(roccurve,
      title = "Receiver Operating Characteristic", legend = true)

    chart1.show()

    logger.log(Priority.INFO, "Generating PR Plot")
    val chart2 = XYAreaChart(prcurve,
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
    fMeasureByThreshold().map((c) => c._2).max,
    areaUnderROC())
}
