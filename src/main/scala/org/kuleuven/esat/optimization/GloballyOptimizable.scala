package org.kuleuven.esat.optimization

import breeze.linalg.DenseVector

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
   *
   * */
  var hyper_parameters: List[String]

  /**
   * Calculates the energy of the configuration,
   * in most global optimization algorithms
   * we aim to find an approximate value of
   * the hyper-parameters such that this function
   * is minimized.
   *
   * @param h The value of the hyper-parameters in the configuration space
   * @return Configuration Energy E(h)
   * */
  def energy(h: DenseVector[Double]): Double

}
