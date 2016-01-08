package io.github.mandar2812.dynaml.optimization

import breeze.linalg.DenseVector
import org.apache.log4j.Logger
import io.github.mandar2812.dynaml.utils

/**
 * @author mandar2812 datum 24/6/15.
 *
 * An implementation of Grid Search
 * global optimization for Kernel Models
 */
class GridSearch[M <: GloballyOptimizable](model: M)
  extends GlobalOptimizer[M]{

  protected val logger = Logger.getLogger(this.getClass)

  override val system = model

  protected var step: Double = 0.3

  protected var gridsize: Int = 3

  protected var logarithmicScale = false

  def setLogScale(t: Boolean) = {
    logarithmicScale = t
    this
  }

  def setGridSize(s: Int) = {
    this.gridsize = s
    this
  }

  def setStepSize(s: Double) = {
    this.step = s
    this
  }

  override def optimize(initialConfig: Map[String, Double],
                        options: Map[String, String] = Map()) = {

    //create grid

    //one list for each key in initialConfig
    val hyper_params = initialConfig.keys.toList

    def scaleFunc(param: String) = if(logarithmicScale)
      (i: Int) => {initialConfig(param)*math.exp((i+1).toDouble*step)}
    else
      (i: Int) => initialConfig(param) - (i+1).toDouble*step

    val gridvecs = initialConfig.map((keyValue) => {
      (keyValue._1, List.tabulate(gridsize)(scaleFunc(keyValue._1)))
    })

    val grid = utils.combine(gridvecs.values).map(x => DenseVector(x.toArray))

    val energyLandscape = grid.map((config) => {
      val configMap = List.tabulate(config.length){i => (hyper_params(i), config(i))}.toMap
      logger.info("Evaluating Configuration: "+configMap)

      val configEnergy = system.energy(configMap, options)

      logger.info("Energy = "+configEnergy+"\n")

      (configEnergy, configMap)
    }).toMap

    val optimum = energyLandscape.keys.min

    logger.info("Optimum value of energy is: "+optimum+
      "\nConfiguration: "+energyLandscape(optimum))

    system.energy(energyLandscape(optimum), options)
    (system, energyLandscape(optimum))
  }
}
