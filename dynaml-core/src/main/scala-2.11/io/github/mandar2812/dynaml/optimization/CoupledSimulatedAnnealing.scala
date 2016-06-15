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

import breeze.stats.distributions.CauchyDistribution

import scala.util.Random

/**
 * @author mandar datum 25/6/15.
 *
 * Implementation of the Coupled Simulated Annealing algorithm
 * for global optimization.
 */
class CoupledSimulatedAnnealing[M <: GloballyOptimizable](model: M)
  extends GridSearch[M](model: M){

  protected var MAX_ITERATIONS: Int = 10

  def setMaxIterations(m: Int) = {
    MAX_ITERATIONS = m
    this
  }

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

  protected val acceptance = (energy: Double,
                              coupling: Double,
                              temperature: Double) => {
    val prob = math.exp(-1.0*energy/temperature)
    prob/(prob+coupling)
  }

  protected val mutate = (config: Map[String, Double], temperature: Double) => {
    config.map((param) => {
      val dist = new CauchyDistribution(0.0, temperature)
      val mutated = param._2 + dist.sample()
      (param._1, math.abs(mutated))
    })
  }

  def acceptanceTemperature(initialTemp: Double)(k: Int): Double =
    initialTemp/math.log(k.toDouble+1.0)

  def mutationTemperature(initialTemp: Double)(k: Int): Double =
    initialTemp/k.toDouble

  override def optimize(initialConfig: Map[String, Double],
                        options: Map[String, String] = Map()) = {

    //create grid
    val iTemp = 1.0
    var accTemp = iTemp
    var mutTemp = iTemp

    val energyLandscape = getEnergyLandscape(initialConfig, options)

    var currentEnergyLandscape = energyLandscape
    var newEnergyLandscape = energyLandscape

    (1 to MAX_ITERATIONS).foreach((iteration) => {
      logger.info("**************************")
      logger.info("CSA Iteration: "+iteration)
      //mutate each element of the grid with
      //the generating distribution
      //and accept using the acceptance distribution
      mutTemp = mutationTemperature(iTemp)(iteration)
      accTemp = acceptanceTemperature(iTemp)(iteration)
      val couplingFactor = currentEnergyLandscape.map(c => math.exp(-1.0*c._1/accTemp)).sum
      //Now mutate each solution and accept/reject
      //according to the acceptance probability

      newEnergyLandscape = currentEnergyLandscape.map((config) => {
        //mutate this config
        val new_config = mutate(config._2, mutTemp)
        val new_energy = system.energy(new_config, options)
        val ans = if(new_energy < config._1) {
          (new_energy, new_config)
        } else {
          val acc = acceptance(new_energy, couplingFactor, accTemp)
          if(Random.nextDouble <= acc) (new_energy, new_config) else config
        }
        ans
      })

      currentEnergyLandscape = newEnergyLandscape

    })

    val landscape = currentEnergyLandscape.toMap
    val optimum = landscape.keys.min

    logger.info("Optimum value of energy is: "+optimum+
      "\nConfiguration: "+landscape(optimum))

    system.energy(landscape(optimum), options)
    (system, landscape(optimum))
  }
}
