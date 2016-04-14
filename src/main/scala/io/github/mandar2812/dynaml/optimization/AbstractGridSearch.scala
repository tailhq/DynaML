/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
* */
package io.github.mandar2812.dynaml.optimization

import breeze.linalg.DenseVector
import org.apache.log4j.Logger
import io.github.mandar2812.dynaml.utils

/**
  * @author mandar2812 datum 01/12/15.
  *
  * An implementation of Grid Search
  * global optimization for general models
  */
class AbstractGridSearch[M <: GloballyOptimizable](model: M)
  extends GlobalOptimizer[M]{

  override protected val logger = Logger.getLogger(this.getClass)

  override val system = model

  override def setLogScale(t: Boolean) = {
    logarithmicScale = t
    this
  }

  override def setGridSize(s: Int) = {
    this.gridsize = s
    this
  }

  override def setStepSize(s: Double) = {
    this.step = s
    this
  }

  override def optimize(initialConfig: Map[String, Double],
                        options: Map[String, String] = Map()) = {

    //create grid

    //one list for each key in initialConfig
    val hyper_params = initialConfig.keys.toList
    val scaleFunc = if(logarithmicScale) (i: Int) => math.exp((i+1).toDouble*step) else
      (i: Int) => (i+1).toDouble*step

    val gridvecs = initialConfig.map((keyValue) => {
      (keyValue._1, List.tabulate(gridsize)(scaleFunc))
    })

    val grid = utils.combine(gridvecs.map(_._2)).map(x => DenseVector(x.toArray))

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

object AbstractGridSearch {
  def apply[M <: GloballyOptimizable](model: M,
               initialConfig: Map[String, Double],
               options: Map[String, String] = Map()): M = {
    new AbstractGridSearch[M](model).optimize(initialConfig, options)._1
  }
}
