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

import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.exp
import io.github.mandar2812.dynaml.DynaMLPipe
import io.github.mandar2812.dynaml.kernels.DecomposableCovariance
import io.github.mandar2812.dynaml.models.gp.AbstractGPRegressionModel
import io.github.mandar2812.dynaml.pipes.DataPipe
import org.apache.log4j.Logger

import scala.reflect.ClassTag

/**
 * @author mandar2812 datum 24/6/15.
 *
 * An implementation of Grid Search
 * global optimization for Kernel Models
 */
class GridSearch[M <: GloballyOptimizable](model: M)
  extends GlobalOptimizer[M]{

  override protected val logger: Logger = Logger.getLogger(this.getClass)

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
      "\nConfiguration: \n"+GlobalOptimizer.prettyPrint(energyLandscape(optimum)))

    //Persist the current configuration to the model memory
    if(options.contains("persist") && (options("persist") == "true" || options("persist") == "1"))
      system.persist(energyLandscape(optimum))

    (system, energyLandscape(optimum))
  }
}

/**
  * @author mandar2812 date: 08/02/2017
  *
  * Performs grid search with a GP model and constructs a probabilistic GP committee.
  * */
class GridGPCommittee[T, I: ClassTag](model: AbstractGPRegressionModel[T, I]) extends GridSearch(model) {

  override protected val logger: Logger = Logger.getLogger(this.getClass)

  override def optimize(initialConfig: Map[String, Double], options: Map[String, String]) = {


    //Find out the blocked hyper parameters and their values
    val blockedHypParams = system.covariance.blocked_hyper_parameters ++ system.noiseModel.blocked_hyper_parameters

    val blockedState = system._current_state.filterKeys(blockedHypParams.contains)

    val energyLandscape = getEnergyLandscape(initialConfig, options)

    //Calculate the weights of each configuration
    val weights = GridGPCommittee.calculateModelWeights(energyLandscape).map(c => (c._1, c._2 ++ blockedState))

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

    //Declare implicit value for weighted kernel
    implicit val encoder = DynaMLPipe.genericReplicationEncoder(weights.length)
    //Declare implicit value for transformation required for creation of a GP model
    implicit val transform: DataPipe[T, Seq[(I, Double)]] = DataPipe(system.dataAsSeq)

    //Now construct a weighted Gaussian Process model
    val (covariancePipe, noisePipe) = (system.covariance.asPipe, system.noiseModel.asPipe)

    //The mean function of the original GP
    val meanF = system.mean

    //Get the kernels, noise models and mean functions of each GP in committee
    val (kernels, noiseModels, meanFuncs) = weights.map(weightCouple => {
      val (w, conf) = weightCouple
      val (k, n) = (covariancePipe(conf), noisePipe(conf))
      (k*w*w, n*w*w, meanF > DataPipe((x: Double) => x*w))
    }).unzip3

    //Calculate the resultant kernels, noise and mean function of GP committee
    val (netKernel, netNoiseModel, netMeanFunc) = (
      new DecomposableCovariance[I](kernels:_*),
      new DecomposableCovariance[I](noiseModels:_*),
      DataPipe((x: I) => meanFuncs.map(_(x)).sum))


    logger.info("Constructing probabilistic Committee GP model")
    //Create the GP committee with the calculated specifications
    val committeeGP: AbstractGPRegressionModel[T, I] =
      AbstractGPRegressionModel(
        netKernel, netNoiseModel, netMeanFunc)(
        system.data, system.npoints)

    logger.info("State of new model: "+GlobalOptimizer.prettyPrint(committeeGP._current_state))

    if(options.contains("persist") && (options("persist") == "true" || options("persist") == "1"))
      committeeGP.persist(committeeGP._current_state)

    //Return the resultant model
    (committeeGP, committeeGP._current_state)
  }
}

object GridGPCommittee {

  def calculateModelWeights(energyLandscape: List[(Double, Map[String, Double])])
  : Seq[(Double, Map[String, Double])] = {

    val h = DenseVector(energyLandscape.map(_._1).toArray)

    val hTotal = sum(h)

    val alpha = if(h.length == 1) 1.0 else 1.0/(h.length-1)

    val weights: DenseVector[Double] = h.map(p => 1.0-(p/hTotal)):*alpha
    val configurations = energyLandscape.map(_._2)

    /*val maskMatrices: Seq[DenseMatrix[Double]] =
      (0 until h.length).map(i =>
        DenseMatrix.tabulate[Double](h.length, h.length)((r,s) => {
          if(r == i) 0.0 else if(s == i) -1.0 else 0.0
        })
      )

    val weights = maskMatrices.map(mask => {
      1.0/sum(exp((mask*h) :* -1.0))
    })*/

    weights.toArray.toSeq zip configurations


  }

}
