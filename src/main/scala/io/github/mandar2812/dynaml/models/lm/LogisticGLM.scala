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

package io.github.mandar2812.dynaml.models.lm

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.models.LinearModel
import io.github.mandar2812.dynaml.optimization._

/**
  * Created by mandar on 31/3/16.
  */
class LogisticGLM(data: Stream[(DenseVector[Double], Double)],
                     numPoints: Int,
                     map: (DenseVector[Double]) => DenseVector[Double] =
                     identity[DenseVector[Double]] _)
  extends LinearModel[Stream[(DenseVector[Double], Double)],
    Int, Int, DenseVector[Double], DenseVector[Double], Double,
    Stream[(DenseVector[Double], Double)]] {

  override protected val g = data

  def dimensions = featureMap(data.head._1).length

  override def initParams(): DenseVector[Double] =
    DenseVector.ones[Double](dimensions+1)


  override protected val optimizer: RegularizedOptimizer[Int, DenseVector[Double],
    DenseVector[Double], Double,
    Stream[(DenseVector[Double], Double)]] =
    new GradientDescent(new LogisticGradient, new SquaredL2Updater)

  override protected var params: DenseVector[Double] = initParams()

  override def clearParameters(): Unit = {
    params = initParams()
  }


  /**
    * Learn the parameters
    * of the model which
    * are in a node of the
    * graph.
    *
    **/
  override def learn(): Unit = {
    params = optimizer.optimize(numPoints,
      g.map(point => (featureMap(point._1), point._2)),
      initParams())
  }

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: DenseVector[Double]): Double =
    params dot featureMap(point)

  /*override protected var hyper_parameters: List[String] = List("regularization")

  /**
    * Calculates the energy of the configuration,
    * in most global optimization algorithms
    * we aim to find an approximate value of
    * the hyper-parameters such that this function
    * is minimized.
    *
    * @param h       The value of the hyper-parameters in the configuration space
    * @param options Optional parameters about configuration
    * @return Configuration Energy E(h)
    **/
  override def energy(h: Map[String, Double], options: Map[String, String]): Double = {
    val designMatrix = DenseMatrix.vertcat[Double](
      g.map(point => featureMap(point._1).toDenseMatrix):_*
    )

    val kernelTraining = designMatrix.t*designMatrix
    val trainingLabels = DenseVector(g.map(_._2).toArray)
    val noiseMat = DenseMatrix.eye[Double](dimensions)*h("regularization")

    AbstractGPRegressionModel.logLikelihood(trainingLabels, kernelTraining, noiseMat)
  }

  override protected var current_state: Map[String, Double] =
    Map("regularization" -> 0.001)

  /**
    * Set the model "state" which
    * contains values of its hyper-parameters
    * with respect to the covariance and noise
    * kernels.
    * */
  def setState(s: Map[String, Double]): this.type ={
    this.setRegParam(s("regularization"))
    current_state = Map("regularization" -> s("regularization"))
    this
  }*/
}
