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

import org.apache.log4j.Logger

/**
 * @author mandar2812 datum 24/6/15.
 *
 * An implementation of Grid Search
 * global optimization for Kernel Models
 */
class GridSearch[M <: GloballyOptimizable](model: M)
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

    val energyLandscape = getEnergyLandscape(initialConfig, options).toMap
    val optimum = energyLandscape.keys.min

    logger.info("Optimum value of energy is: "+optimum+
      "\nConfiguration: "+energyLandscape(optimum))

    system.energy(energyLandscape(optimum), options)
    (system, energyLandscape(optimum))
  }
}
