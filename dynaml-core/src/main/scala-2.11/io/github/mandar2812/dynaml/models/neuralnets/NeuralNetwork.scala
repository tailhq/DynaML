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

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.graph.NeuralGraph
import io.github.mandar2812.dynaml.models.ParameterizedLearner
import io.github.mandar2812.dynaml.pipes.{DataPipe, MetaPipe, Scaler}

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
@deprecated("Neural Network base trait has been deprecated since DynaML v1.4.1")
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

/**
  * The basic building block of a neural computation stack.
  *
  * @tparam P The type of the parameters/connections of the layer.
  * @tparam I The type of the input supplied to the layer
  * */
trait NeuralLayer[P, I] extends Scaler[I] {

  /**
    * The layer synapses or connection weights
    * */
  val parameters: P

  /**
    * Activation function
    * */
  val activationFunc: Activation[I]

  /**
    * Compute the forward pass through the layer.
    * */
  val forward: Scaler[I] = this > activationFunc

}

object NeuralLayer {

  def apply[P, I](compute: MetaPipe[P, I, I], activation: Activation[I])(params: P) =
    new NeuralLayer[P, I] {
      override val parameters = params
      override val activationFunc = activation
      override def run(data: I) = activation(compute(parameters)(data))
  }

}

/**
  * A network, represented as a stack of [[NeuralLayer]] objects.
  * */
class NeuralStack[P, I](elements: NeuralLayer[P, I]*) {

  val layers: Seq[NeuralLayer[P, I]] = elements

  val layerWeights = layers.map(_.parameters)

  /**
    * Do a forward pass through the network outputting all the intermediate.
    * layer activations.
    * */
  def forwardPropagate(x: I): Seq[I] = layers.scanLeft(x)((h, layer) => layer.forward(h))

  /**
    * Do a forward pass through the network outputting only the output layer activations.
    * */
  def forwardPass(x: I): I = layers.foldLeft(x)((h, layer) => layer.forward(h))

  /**
    * Slice the stack according to a range.
    * */
  def apply(r: Range): NeuralStack[P, I] = NeuralStack(layers.slice(r.min, r.max + 1):_*)

}

object NeuralStack {

  def apply[P, I](elements: NeuralLayer[P, I]*): NeuralStack[P, I] = new NeuralStack(elements:_*)
}

/**
  * A mechanism to generate neural computation layers on the fly.
  * */
class NeuralLayerFactory[P, I](
  metaLayer: MetaPipe[P, I, I],
  activationFunc: Activation[I]) extends
  DataPipe[P, NeuralLayer[P, I]] {

  override def run(params: P) = NeuralLayer(metaLayer, activationFunc)(params)
}
