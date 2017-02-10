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

import breeze.linalg.{DenseMatrix, DenseVector, max, min, sum}
import breeze.numerics.exp
import breeze.stats.distributions.CauchyDistribution
import io.github.mandar2812.dynaml.kernels.DecomposableCovariance
import io.github.mandar2812.dynaml.models.gp.AbstractGPRegressionModel
import io.github.mandar2812.dynaml.pipes.{DataPipe, WeightedSumReducer}
import io.github.mandar2812.dynaml.{DynaMLPipe, utils}

import scala.reflect.ClassTag
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

  private def computeDesiredVariance = CoupledSimulatedAnnealing.varianceDesired(variant) _

  private def computeCouplingFactor = CoupledSimulatedAnnealing.couplingFactor(variant) _

  private def computeAcceptanceProb = CoupledSimulatedAnnealing.acceptanceProbability(variant) _

  protected val mutate = (config: Map[String, Double], temperature: Double) => {
    logger.info("Mutating configuration: \n"+GlobalOptimizer.prettyPrint(config)+"\n")

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

  protected def performCSA(
    initialConfig: Map[String, Double],
    options: Map[String, String] = Map()) = {

    logger.info("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
    logger.info("Coupled Simulated Annealing: "+CoupledSimulatedAnnealing.algorithm(variant))
    logger.info("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")

    var accTemp = iTemp
    var mutTemp = iTemp

    //Calculate desired variance
    val sigmaD = computeDesiredVariance(math.pow(gridsize, initialConfig.size).toInt)

    val initialEnergyLandscape = getEnergyLandscape(initialConfig, options)

    val gamma_init = computeCouplingFactor(initialEnergyLandscape.map(_._1), accTemp)

    var acceptanceProbs: List[Double] = initialEnergyLandscape.map(c => {
      computeAcceptanceProb(c._1, c._1, gamma_init, accTemp)
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

          val couplingFactor = computeCouplingFactor(eLandscape.map(t => t._1 - maxEnergy), accTemp)

          //Now mutate each solution and accept/reject
          //according to the acceptance probability
          val (newEnergyLandscape,probabilities) = eLandscape.map((config) => {
            //mutate this config
            val new_config = mutate(config._2, mutTemp)
            val new_energy = system.energy(new_config, options)

            logger.info("New Configuration: \n"+GlobalOptimizer.prettyPrint(new_config))
            logger.info("Energy = "+new_energy)

            //Calculate the acceptance probability
            val acceptanceProbability =
              computeAcceptanceProb(
                new_energy - maxEnergy, config._1,
                couplingFactor, accTemp)

            val ans = if(new_energy < config._1) {

              logger.info("Status: Accepted\n")
              ((new_energy, new_config), acceptanceProbability)

            } else {

              if(Random.nextDouble <= acceptanceProbability) {
                logger.info("Status: Accepted\n")
                ((new_energy, new_config), acceptanceProbability)
              } else {
                logger.info("Status: Rejected\n")
                (config, acceptanceProbability)
              }
            }

            ans
          }).unzip

          acceptanceProbs = probabilities
          CSATRec(newEnergyLandscape, it-1)
      }

    CSATRec(initialEnergyLandscape, MAX_ITERATIONS)
  }

  override def optimize(initialConfig: Map[String, Double],
                        options: Map[String, String] = Map()) = {


    val landscape = performCSA(initialConfig, options).toMap
    val optimum = landscape.keys.min

    logger.info(
      "Optimum value of energy is: "+optimum+
      " at: "+GlobalOptimizer.prettyPrint(landscape(optimum)))

    //Persist the current configuration to the model memory
    if(options.contains("persist") && (options("persist") == "true" || options("persist") == "1"))
      system.persist(landscape(optimum))

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

  def couplingFactor(variant: String)(
    landscape: Seq[Double],
    Tacc: Double): Double = {

    if(variant == MuSA || variant == BA)
      landscape.map(energy => math.exp(-1.0*energy/Tacc)).sum
    else if (variant == M || variant == MwVC)
      landscape.map(energy => math.exp(energy/Tacc)).sum
    else 1.0

  }

  def acceptanceProbability(variant: String)(
    energy: Double, oldEnergy: Double,
    gamma: Double, temperature: Double) = {

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

/**
  * @author mandar2812 date 08/02/2017
  *
  * Build GP committee model after performing the CSA routine
  * */
class ProbGPCommMachine[T, I: ClassTag](
  model: AbstractGPRegressionModel[T, I]) extends
  CoupledSimulatedAnnealing(model) {

  private var policy: String = "CSA"

  private var baselinePolicy: String = "max"

  def _policy = policy

  def setPolicy(p: String): this.type = {
    if(p == "CSA" || p == "Coupled Simulated Annealing")
      policy = "CSA"
    else
      policy = "GS"

    this
  }

  def setBaseLinePolicy(p: String): this.type = {

    if(p == "avg" || p == "mean" || p == "average")
      baselinePolicy = "mean"
    else if(p == "min")
      baselinePolicy = "min"
    else if(p == "max")
      baselinePolicy = "max"
    else
      baselinePolicy = "mean"

    this
  }

  private def calculateEnergyLandscape(initialConfig: Map[String, Double], options: Map[String, String]) =
    if(policy == "CSA") performCSA(initialConfig, options)
    else getEnergyLandscape(initialConfig, options)

  private def modelProbabilities = DataPipe(ProbGPCommMachine.calculateModelWeightsSigmoid(baselinePolicy) _)

  override def optimize(
    initialConfig: Map[String, Double],
    options: Map[String, String]) = {


    //Find out the blocked hyper parameters and their values
    val blockedHypParams = system.covariance.blocked_hyper_parameters ++ system.noiseModel.blocked_hyper_parameters

    val (kernelParams, noiseParams) = (
      system.covariance.hyper_parameters,
      system.noiseModel.hyper_parameters)

    val blockedState = system._current_state.filterKeys(blockedHypParams.contains)

    val energyLandscape = calculateEnergyLandscape(initialConfig, options)

    //Calculate the weights of each configuration
    val weights = modelProbabilities(energyLandscape).map(c => (c._1, c._2 ++ blockedState))

    //Declare implicit value for weighted kernel
    implicit val encoder = DynaMLPipe.genericReplicationEncoder(weights.length)
    //Declare implicit value for transformation required for creation of the compound kernel
    implicit val transform: DataPipe[T, Seq[(I, Double)]] = DataPipe(system.dataAsSeq)
    //Declare implicit reducer required for the weighted kernel
    implicit val reducer = WeightedSumReducer(weights.map(c => c._1*c._1).toArray)

    //Now construct a weighted Gaussian Process model
    val (covariancePipe, noisePipe) = (system.covariance.asPipe, system.noiseModel.asPipe)

    //The mean function of the original GP
    val meanF = system.mean

    //Get the kernels, noise models and mean functions of each GP in the committee
    val (kernels, noiseModels, meanFuncs) = weights.map(weightCouple => {

      val (w, conf) = weightCouple

      val (k, n) = (
        covariancePipe(conf.filterKeys(kernelParams.contains)),
        noisePipe(conf.filterKeys(noiseParams.contains)))

      (k, n, meanF > DataPipe((x: Double) => x*w))
    }).unzip3

    //Calculate the resultant kernels, noise and mean function of GP committee
    val (netKernel, netNoiseModel, netMeanFunc) = (
      new DecomposableCovariance[I](kernels:_*),
      new DecomposableCovariance[I](noiseModels:_*),
      DataPipe((x: I) => meanFuncs.map(_(x)).sum))

    logger.info("===============================================")
    logger.info("Constructing probabilistic Committee GP model")
    //Create the GP committee with the calculated specifications
    val committeeGP: AbstractGPRegressionModel[T, I] =
      AbstractGPRegressionModel(
        netKernel, netNoiseModel, netMeanFunc)(
        system.data, system.npoints)

    logger.info("Number of model instances = "+weights.length)
    logger.info("--------------------------------------")
    logger.info(
      "Calculated model probabilities/weights are \n"+
        weights.map(wc =>
          "\nConfiguration: \n"+
            GlobalOptimizer.prettyPrint(wc._2)+
            "\nProbability = "+wc._1+"\n"
        ).reduceLeft((a, b) => a++b)
    )
    logger.info("--------------------------------------")


    logger.info(
      "State of new model:- Covariance: \n" +
        GlobalOptimizer.prettyPrint(committeeGP.covariance.state))
    logger.info(
      "State of new model:- Noise \n" +
        GlobalOptimizer.prettyPrint(committeeGP.noiseModel.state))

    logger.info("===============================================")

    if(options.contains("persist") && (options("persist") == "true" || options("persist") == "1")) {
      logger.info("Persisting model state")
      committeeGP.persist(committeeGP._current_state)
    }

    //Return the resultant model
    (committeeGP, committeeGP._current_state)
  }
}

object ProbGPCommMachine {

  def calculateModelWeights(energyLandscape: List[(Double, Map[String, Double])])
  : Seq[(Double, Map[String, Double])] = {

    val h = DenseVector(energyLandscape.map(_._1).toArray)

    val hTotal = sum(h)

    val alpha = if (h.length == 1) 1.0 else 1.0 / (h.length - 1)

    val weights: DenseVector[Double] = h.map(p => 1.0 - (p / hTotal)) :* alpha
    val configurations = energyLandscape.map(_._2)

    weights.toArray.toSeq zip configurations
  }

  def calculateModelWeightsSigmoid(
    baselineMethod: String = "mean")(
    energyLandscape: List[(Double, Map[String, Double])])
  : Seq[(Double, Map[String, Double])] = {

    val h = DenseVector(energyLandscape.map(_._1).toArray)

    val baseline = baselineMethod match {
      case "mean" =>
        sum(h)/h.length.toDouble
      case "avg" =>
        sum(h)/h.length.toDouble
      case "min" =>
        min(h)
      case "max" =>
        max(h)
      case _ =>
        sum(h)/h.length.toDouble
    }

    val maskMatrices: Seq[DenseMatrix[Double]] =
      (0 until h.length).map(i =>
        DenseMatrix.tabulate[Double](h.length, h.length)((r, s) => {
          if (r == i) 0.0
          else if (s == i) -1.0
          else if (r == s) 1.0
          else 0.0
        })
      )

    val weights = maskMatrices.zipWithIndex.map(mask => {
      1.0/sum(exp((mask._1*h) :* (-1.0/baseline)))
    })

    val configurations = energyLandscape.map(_._2)

    weights.toArray.toSeq zip configurations

  }


}