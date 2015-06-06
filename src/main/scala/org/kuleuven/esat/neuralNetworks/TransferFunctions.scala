package org.kuleuven.esat.neuralNetworks

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
  val logsig = sigmoid _

  /**
   * Identity Function
   * */
  val lin = identity _

  /**
   * Function which returns
   * the appropriate activation
   * for a given string
   * */
  val getActivation = (func: String) =>
    func match {
      case "sigmoid" => logsig
      case "logsig" => logsig
      case "tansig" => tansig
      case "linear" => lin
    }
}
