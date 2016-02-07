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

  /**
   * Sigmoid/Logistic function
   *
   * */
  val logsig = (x:Double) => sigmoid(x)

  /**
   * Identity Function
   * */
  val lin = (x: Double) => identity(x)

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
    }
}
