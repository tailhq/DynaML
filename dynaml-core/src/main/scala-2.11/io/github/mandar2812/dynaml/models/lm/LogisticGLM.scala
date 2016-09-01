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
import breeze.numerics._
import io.github.mandar2812.dynaml.optimization._
import org.apache.commons.math3.distribution.NormalDistribution

/**
  * @author mandar2812 date: 31/3/16.
  *
  * Logistic model for binary classification.
  * @param data The training data as a stream of tuples
  * @param numPoints The number of training data points
  * @param map The basis functions used to map the input
  *            features to a possible higher dimensional space
  */
class LogisticGLM(data: Stream[(DenseVector[Double], Double)],
                  numPoints: Int,
                  map: (DenseVector[Double]) => DenseVector[Double] =
                  identity[DenseVector[Double]] _)
  extends GeneralizedLinearModel[
    Stream[(DenseVector[Double], Double)]
    ](data, numPoints, map) {

  override val h: (Double) => Double = (x) => sigmoid(x)

  override val task = "classification"

  override protected val optimizer: RegularizedOptimizer[DenseVector[Double],
    DenseVector[Double], Double,
    Stream[(DenseVector[Double], Double)]] =
    new GradientDescent(new LogisticGradient, new SquaredL2Updater)

  override def prepareData(d: Stream[(DenseVector[Double], Double)]) =
    d.map(point => (featureMap(point._1), point._2))

}

/**
  * Probit model for binary classification.
  * @param data The training data as a stream of tuples
  * @param numPoints The number of training data points
  * @param map The basis functions used to map the input
  *            features to a possible higher dimensional space
  */
class ProbitGLM(data: Stream[(DenseVector[Double], Double)],
                numPoints: Int,
                map: (DenseVector[Double]) => DenseVector[Double] =
                identity[DenseVector[Double]] _)
  extends LogisticGLM(data, numPoints, map) {

  private val standardGaussian = new NormalDistribution(0, 1.0)

  override val h = (x: Double) =>
    standardGaussian.cumulativeProbability(x)

  override protected val optimizer =
    new GradientDescent(
      new ProbitGradient,
      new SquaredL2Updater)

}