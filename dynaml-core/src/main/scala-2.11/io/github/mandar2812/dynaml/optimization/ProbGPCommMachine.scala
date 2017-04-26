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
import io.github.mandar2812.dynaml.DynaMLPipe
import io.github.mandar2812.dynaml.kernels.DecomposableCovariance
import io.github.mandar2812.dynaml.models.gp.AbstractGPRegressionModel
import io.github.mandar2812.dynaml.pipes.{DataPipe, WeightedSumReducer}

import scala.reflect.ClassTag

/**
  * Build GP committee model after performing the CSA routine
  *
  * @author mandar2812 date 08/02/2017
  *
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
    else getEnergyLandscape(initialConfig, options, meanFieldPrior)

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