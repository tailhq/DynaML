package org.kuleuven.esat.optimization

/**
 * @author mandar2812, datum: 23/6/15.
 *
 * We define a common binding
 * characteristic between all "globally optimizable"
 * models i.e. models where hyper-parameters can
 * be optimized/tuned.
 */
trait GloballyOptimizable {

  /**
   * Stores the names of the hyper-parameters
   * */
  protected var hyper_parameters: List[String]

  /**
   * A Map which stores the current state of
   * the system.
   * */
  protected var current_state: Map[String, Double]

  /**
   * Calculates the energy of the configuration,
   * in most global optimization algorithms
   * we aim to find an approximate value of
   * the hyper-parameters such that this function
   * is minimized.
   *
   * @param h The value of the hyper-parameters in the configuration space
   * @param options Optional parameters about configuration
   * @return Configuration Energy E(h)
   * */
  def energy(h: Map[String, Double],
             options: Map[String, AnyRef] = Map()): Double

}
