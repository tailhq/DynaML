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
package io.github.mandar2812.dynaml.models.neuralnets

import breeze.linalg.{DenseVector, DenseMatrix}
import io.github.mandar2812.dynaml.models.ParameterizedLearner

/**
  *
  * Top level trait defining
  * properties of a neural network.
  *
  * @tparam G The type of the data structure containing the
  *           training data set.
  *
  * @tparam P The underlying base graph from which the [[NeuralGraph]]
  *           object is constructed.
  *
  * @tparam T A subclass of the [[NeuralGraph]] object with [[P]] as the
  *           base graph
  *
  * @tparam Pattern The type of an individual data pattern
  * */
trait NeuralNetwork[G, P, T <: NeuralGraph[P], Pattern] extends
ParameterizedLearner[G, T,
  DenseVector[Double], DenseVector[Double],
  Stream[Pattern]] {

  val inputDimensions: Int

  val outputDimensions: Int

  val hiddenLayers: Int

  val activations: List[(Double) => Double]

  val neuronCounts: List[Int]

  /**
    * Convert the data structure from type [[G]]
    * to a [[Stream]] of [[Pattern]] objects
    *
    * */
  def dataAsStream(d: G): Stream[Pattern]
}