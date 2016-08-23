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
import breeze.stats.distributions.CauchyDistribution
import io.github.mandar2812.dynaml.utils

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

  protected var variant = CoupledSimulatedAnnealing.MuSA

  def setVariant(v: String) = {
    variant = v
    this
  }

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

  var iTemp = 1.0

  var alpha = 0.05

  protected val mutate = (config: Map[String, Double], temperature: Double) => {
    logger.info("Mutating configuration: "+GlobalOptimizer.prettyPrint(config))
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

    logger.info("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
    logger.info("Coupled Simulated Annealing: "+CoupledSimulatedAnnealing.algorithm(variant))
    logger.info("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")

    var accTemp = iTemp
    var mutTemp = iTemp

    //Calculate desired variance
    val sigmaD = CoupledSimulatedAnnealing.varianceDesired(variant)(math.pow(gridsize, initialConfig.size).toInt)

    val initialEnergyLandscape = getEnergyLandscape(initialConfig, options)


    val gamma_init = CoupledSimulatedAnnealing.couplingFactor(variant)(initialEnergyLandscape.map(_._1), accTemp)
    var acceptanceProbs: List[Double] = initialEnergyLandscape.map(c => {
      CoupledSimulatedAnnealing.acceptanceProbability(variant)(c._1, c._1, gamma_init, accTemp)
    })

    def CSATRec(eLandscape: List[(Double, Map[String, Double])], it: Int): List[(Double, Map[String, Double])] =
      it match {
        case 0 => eLandscape
        case num =>
          logger.info("**************************")
          logger.info("CSA Iteration: "+(MAX_ITERATIONS-it+1))
          //mutate each element of the grid with
          //the generating distribution
          //and accept using the acceptance distribution
          mutTemp = mutationTemperature(iTemp)(it)
          accTemp = variant match {
            case CoupledSimulatedAnnealing.MwVC =>
              val (_,variance) = utils.getStats(acceptanceProbs.map(DenseVector(_)))

              if (variance(0) < sigmaD)
                accTemp * (1-alpha)
              else accTemp * (1+alpha)
            case _ =>
              acceptanceTemperature(iTemp)(it)
          }

          val maxEnergy = eLandscape.map(_._1).max

          val couplingFactor = CoupledSimulatedAnnealing.couplingFactor(variant)(
            eLandscape.map(t => t._1 - maxEnergy),
            accTemp)

          //Now mutate each solution and accept/reject
          //according to the acceptance probability
          val (newEnergyLandscape,probabilities) = eLandscape.map((config) => {
            //mutate this config
            val new_config = mutate(config._2, mutTemp)
            val new_energy = system.energy(new_config, options)

            //Calculate the acceptance probability
            val acceptanceProbability = CoupledSimulatedAnnealing.acceptanceProbability(variant)(
              new_energy - maxEnergy, config._1,
              couplingFactor, accTemp)

            val ans = if(new_energy < config._1) {
              ((new_energy, new_config), acceptanceProbability)
            } else {
              if(Random.nextDouble <= acceptanceProbability) ((new_energy, new_config), acceptanceProbability)
              else (config, acceptanceProbability)
            }
            ans
          }).unzip
          acceptanceProbs = probabilities
          CSATRec(newEnergyLandscape, it-1)
      }

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

  def algorithm(variant: String): String = variant match {
    case MuSA => "Multi-state Simulated Annealing"
    case BA => "Blind Acceptance"
    case M => "Modified"
    case MwVC => "Modified with Variance Control"
  }

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

  def varianceDesired(variant: String)(m: Int):Double = {
    if(variant == MuSA || variant == BA)
      0.99
    else
      0.99*(m-1)/math.pow(m, 2.0)

  }


}