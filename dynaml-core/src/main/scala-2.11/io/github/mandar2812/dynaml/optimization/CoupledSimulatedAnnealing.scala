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

  var variant = CoupledSimulatedAnnealing.MuSA

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

  protected def acceptance(energy: Double, oldEnergy: Double, coupling: Double, temperature: Double) =
    CoupledSimulatedAnnealing.acceptanceProbability(variant)(energy, oldEnergy, coupling, temperature)

  protected val mutate = (config: Map[String, Double], temperature: Double) => {
    logger.info("Mutating configuration: "+config)
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

  def gamma(landscape: Seq[Double])(accTemp: Double): Double =
    CoupledSimulatedAnnealing.couplingFactor(variant)(landscape, accTemp)

  override def optimize(initialConfig: Map[String, Double],
                        options: Map[String, String] = Map()) = {

    //create grid
    val iTemp = 2.0
    var accTemp = iTemp
    var mutTemp = iTemp

    val initialEnergyLandscape = getEnergyLandscape(initialConfig, options)

    def CSATRec(eLandscape: Seq[(Double, Map[String, Double])], it: Int): Seq[(Double, Map[String, Double])] =
      it match {
        case 0 => eLandscape
        case num =>
          logger.info("**************************")
          logger.info("CSA Iteration: "+(MAX_ITERATIONS-it+1))
          //mutate each element of the grid with
          //the generating distribution
          //and accept using the acceptance distribution
          mutTemp = mutationTemperature(iTemp)(it)
          accTemp = acceptanceTemperature(iTemp)(it)

          val maxEnergy = eLandscape.map(_._1).max

          val couplingFactor = gamma(eLandscape.map(t => t._1 - maxEnergy))(accTemp)
          //Now mutate each solution and accept/reject
          //according to the acceptance probability
          val newEnergyLandscape = eLandscape.map((config) => {
            //mutate this config
            val new_config = mutate(config._2, mutTemp)
            val new_energy = system.energy(new_config, options)
            val ans = if(new_energy < config._1) {
              (new_energy, new_config)
            } else {
              val acc = acceptance(new_energy - maxEnergy, config._1, couplingFactor, accTemp)
              if(Random.nextDouble <= acc) (new_energy, new_config) else config
            }
            ans
          })
          CSATRec(newEnergyLandscape, it-1)
      }


    /*cfor(1)(iteration => iteration <= MAX_ITERATIONS, iteration => iteration + 1)( iteration => {
      logger.info("**************************")
      logger.info("CSA Iteration: "+iteration)
      //mutate each element of the grid with
      //the generating distribution
      //and accept using the acceptance distribution
      mutTemp = mutationTemperature(iTemp)(iteration)
      accTemp = acceptanceTemperature(iTemp)(iteration)

      val maxEnergy = currentEnergyLandscape.map(_._1).max

      val couplingFactor = gamma(currentEnergyLandscape.map(t => t._1 - maxEnergy))(accTemp)
      //Now mutate each solution and accept/reject
      //according to the acceptance probability
      val newEnergyLandscape = currentEnergyLandscape.map((config) => {
        //mutate this config
        val new_config = mutate(config._2, mutTemp)
        val new_energy = system.energy(new_config, options)
        val ans = if(new_energy < config._1) {
          (new_energy, new_config)
        } else {
          val acc = acceptance(new_energy - maxEnergy, config._1, couplingFactor, accTemp)
          if(Random.nextDouble <= acc) (new_energy, new_config) else config
        }
        ans
      })

      currentEnergyLandscape = newEnergyLandscape

    })*/

    val landscape = CSATRec(initialEnergyLandscape, MAX_ITERATIONS).toMap
    val optimum = landscape.keys.min

    logger.info("Optimum value of energy is: "+optimum+
      "\nConfiguration: "+landscape(optimum))

    system.energy(landscape(optimum), options)
    (system, landscape(optimum))
  }
}


object CoupledSimulatedAnnealing {

  val MuSA = "CSA-MuSA"
  val BA = "CSA-BA"
  val M = "CSA-M"
  val MwVC = "CSA-MwVC"
  val SA = "SA"

  def couplingFactor(variant: String)(landscape: Seq[Double], Tacc: Double): Double = {
    if(variant == MuSA || variant == BA)
      landscape.map(energy => math.exp(-1.0*energy/Tacc)).sum
    else if (variant == M || variant == MwVC)
      landscape.map(energy => math.exp(energy/Tacc)).sum
    else 1.0
  }

  def acceptanceProbability(variant: String)(energy: Double, oldEnergy: Double, gamma: Double, temperature: Double) = {
    if(variant == MuSA )
      math.exp(-1.0*energy/temperature)/(math.exp(-1.0*energy/temperature)+gamma)
    else if (variant == BA)
      1.0 - (math.exp(-1.0*oldEnergy/temperature)/gamma)
    else if (variant == M || variant == MwVC)
      math.exp(oldEnergy/temperature)/gamma
    else gamma/(1.0 + math.exp((energy - oldEnergy)/temperature))
  }


}