package io.github.mandar2812.dynaml.evaluation

import breeze.linalg.DenseVector
import org.apache.log4j.{Priority, Logger}
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

  val rmse: Double = math.sqrt(scores.map((p) =>
    math.pow(p._1 - p._2, 2)/length).sum)

  val mae: Double = scores.map((p) =>
    math.abs(p._1 - p._2)/length).sum

  val rmsle: Double = math.sqrt(scores.map((p) =>
    math.pow(math.log(1 + math.abs(p._1)) - math.log(math.abs(p._2) + 1),
      2)/length).sum)

  val Rsq: Double = RegressionMetricsSpark.computeRsq(scores, length)

  def residuals() = this.scores.map((s) => (s._1 - s._2, s._2))

  def scores_and_labels() = this.scoresAndLabels

  override def print(): Unit = {
    logger.log(Priority.INFO, "Regression Model Performance")
    logger.log(Priority.INFO, "============================")
    logger.log(Priority.INFO, "MAE: " + mae)
    logger.log(Priority.INFO, "RMSE: " + rmse)
    logger.log(Priority.INFO, "RMSLE: " + rmsle)
    logger.log(Priority.INFO, "R^2: " + Rsq)
  }

  override def kpi() = DenseVector(mae, rmse, Rsq)

  override def generatePlots(): Unit = {
    implicit val theme = org.jfree.chart.StandardChartTheme.createDarknessTheme
    val roccurve = this.residuals().map(c => (c._2, c._1)).collect().toList

    logger.log(Priority.INFO, "Generating Plot of Residuals")
    val chart1 = XYBarChart(roccurve,
      title = "Residuals", legend = true)

    chart1.show()
  }

}

object RegressionMetricsSpark {
  def computeRsq(scoresAndLabels: RDD[(Double, Double)], size: Long): Double = {

    val mean: Double = scoresAndLabels.map{coup => coup._2}.sum/size
    var SSres = 0.0
    var SStot = 0.0
    scoresAndLabels.foreach((couple) => {
      SSres += math.pow(couple._2 - couple._1, 2)
      SStot += math.pow(couple._2 - mean, 2)
    })
    1 - (SSres/SStot)
  }
}
