package io.github.mandar2812.dynaml.models.neuralnets

import breeze.numerics.sigmoid

/**
 * @author mandar2812
 *
 * Object implementing the various transfer functions.
 */

object TransferFunctions {

  /**
   * Hyperbolic tangent function
   * */
  val tansig = math.tanh _
  val Dtansig = (x: Double) => tansig(x)/(math.sinh(x)*math.cosh(x))

  /**
   * Sigmoid/Logistic function
   *
   * */
  val logsig = (x:Double) => sigmoid(x)
  val Dlogsig = (x:Double) => sigmoid(x)*(1-sigmoid(x))

  /**
   * Identity Function
   * */
  val lin = (x: Double) => identity(x)
  val Dlin = (_: Double) => 1.0

  val recLin = (x: Double) => math.max(0.0, x)
  val DrecLin = (x: Double) => if(x < 0 ) 0.0 else 1.0

  /**
   * Function which returns
   * the appropriate activation
   * for a given string
   * */
  def getActivation(s: String): (Double) => Double =
    s match {
      case "sigmoid" => logsig
      case "logsig" => logsig
      case "tansig" => tansig
      case "linear" => lin
      case "recLinear" => recLin
    }

  def getDiffActivation(s: String): (Double) => Double =
    s match {
      case "sigmoid" => Dlogsig
      case "logsig" => Dlogsig
      case "tansig" => Dtansig
      case "linear" => Dlin
      case "recLinear" => DrecLin
    }
}
