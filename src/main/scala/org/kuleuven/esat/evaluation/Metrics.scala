package org.kuleuven.esat.evaluation

/**
 * Abstract trait for metrics
 */
trait Metrics[P] {
  protected val scoresAndLabels: List[(P, P)]
  def print(): Unit
  def generatePlots(): Unit = {}
}

object Metrics{
  def apply(task: String)
           (scoresAndLabels: List[(Double, Double)])
  : Metrics[Double] = task match {
    case "regression" => new RegressionMetrics(scoresAndLabels)
    case "classification" => new BinaryClassificationMetrics(scoresAndLabels)
  }
}
