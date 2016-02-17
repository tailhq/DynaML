package io.github.mandar2812.dynaml.models.neuralnets

import breeze.numerics.sigmoid

/**
 * @author mandar2812
 *
 * Object implementing neural network transfer functions.
 */

object TransferFunctions {

  /**
   * Hyperbolic tangent function
   * */
  val tansig = math.tanh _

  /**
    * First derivative of the hyperbolic tangent function
    * */
  val Dtansig = (x: Double) => tansig(x)/(math.sinh(x)*math.cosh(x))

  /**
   * Sigmoid/Logistic function
   *
   * */
  val logsig = (x:Double) => sigmoid(x)

  /**
    * First derivative of the sigmoid function
    * */
  val Dlogsig = (x:Double) => sigmoid(x)*(1-sigmoid(x))

  /**
   * Linear/Identity Function
   * */
  val lin = (x: Double) => identity(x)

  /**
    * First derivative of the linear function
    * */
  val Dlin = (_: Double) => 1.0

  /**
    * Rectified linear Function
    * */
  val recLin = (x: Double) => math.max(0.0, x)

  /**
    * First derivative of the rectified linear function
    * */
  val DrecLin = (x: Double) => if(x < 0 ) 0.0 else 1.0

  /**
   * Function which returns
   * the appropriate activation
   * for a given string
   * */
  def getActivation(s: String): (Double) => Double =
    s.toLowerCase match {
      case "sigmoid" => logsig
      case "logsig" => logsig
      case "tansig" => tansig
      case "linear" => lin
      case "reclinear" => recLin
    }

  /**
    * Function which returns
    * the appropriate derivative
    * activation for a given string
    * */
  def getDiffActivation(s: String): (Double) => Double =
    s.toLowerCase match {
      case "sigmoid" => Dlogsig
      case "logsig" => Dlogsig
      case "tansig" => Dtansig
      case "linear" => Dlin
      case "reclinear" => DrecLin
    }
}
