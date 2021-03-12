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
package io.github.tailhq.dynaml.models.neuralnets

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.tailhq.dynaml.graph.FFNeuralGraph
import io.github.tailhq.dynaml.graph.utils.Neuron
import io.github.tailhq.dynaml.optimization.{BackPropagation, FFBackProp, GradBasedBackPropagation}
import io.github.tailhq.dynaml.pipes.{ReversibleScaler, Scaler}
import io.github.tailhq.dynaml.probability.RandomVariable

/**
  * Base implementation of a Sparse Autoencoder
  *
  * It is represented as a [[ReversibleScaler]] transforming
  * a breeze [[DenseVector]] into another breeze Dense Vector.
  *
  * @author tailhq 22/6/16.
  *
  * */
@deprecated("AutoEncoder is deprecated as of DynaML v1.4.2 in favour of GenericAutoEncoder")
class AutoEncoder(inDim: Int, outDim: Int,
                  acts: List[String] = List("logsig", "logsig"))
  extends ReversibleScaler[DenseVector[Double]]{

  def initialize() =
    FFNeuralGraph(
      inDim, inDim, 1, acts,
      List(outDim))

  var graph = initialize()

  val optimizer = new BackPropagation

  val i = Scaler((xhat: DenseVector[Double]) => {
    graph.getLayer(1)
      .filter(_.getNeuronType() == "perceptron")
      .foreach(n => n.setValue(xhat(n.getNID()-1)))

    val outputs:Map[Int, Double] = graph.getLayer(2).filter(_.getNeuronType() == "perceptron")
      .map(outputNeuron => (outputNeuron.getNID(), Neuron.getLocalField(outputNeuron)._1))
      .toMap

    DenseVector.tabulate[Double](inDim)(i => outputs(i+1))
  })

  def learn(data: Stream[(DenseVector[Double], DenseVector[Double])]) = {
    graph = optimizer.optimize(data.length.toLong, data, initialize())
  }

  override def run(x: DenseVector[Double]) = {
    graph.forwardPass(x)

    val outputs:Map[Int, Double] = graph.getLayer(1).filter(_.getNeuronType() == "perceptron")
      .map(outputNeuron => (outputNeuron.getNID(), Neuron.getLocalField(outputNeuron)._1))
      .toMap

    DenseVector.tabulate[Double](outDim)(i => outputs(i+1))

  }

}

/**
  * Represents an auto-encoder acting on generic types.
  *
  * @tparam LayerP The type of parameters specifying layer weights
  * @tparam I The type of input features
  * @author tailhq date: 29/03/17
  *
  * */
class GenericAutoEncoder[LayerP, I](
  learningAlgorithm: GradBasedBackPropagation[LayerP, I],
  initializer: RandomVariable[Seq[LayerP]]) {

  val optimizer = learningAlgorithm

  val stackFactory = optimizer.stackFactory

  private val nLayers: Int = stackFactory.layerFactories.length

  /**
    * The depth of the encoder is the number of
    * layers in the forward transform, this is
    * calculated in a straight forward manner by
    * dividing the total number of layers by 2.
    * */
  val depth: Int = nLayers/2

  val generator: RandomVariable[Seq[LayerP]] = initializer

  def initialize: NeuralStack[LayerP, I] = (generator.sample > stackFactory).run()

  var stack: NeuralStack[LayerP, I] = initialize

  /**
    * Learn a representation from data.
    * */
  def learn(data: Stream[I]): Unit = stack = optimizer.optimize(data.length, data.map(x => (x,x)), initialize)

  /**
    * @return A slice of the total [[NeuralStack]] which represents the encoding transformation.
    * */
  def forwardStack: NeuralStack[LayerP, I] = stack(0 until depth)

  /**
    * @return A slice of the total [[NeuralStack]] which represents the decoding transformation.
    * */
  def reverseStack: NeuralStack[LayerP, I] = stack(depth until nLayers)

  /**
    * Encode a point
    * @param x The point to be encoded
    *
    * */
  def f(x: I): I = forwardStack.forwardPass(x)

  /**
    * Decode a point
    * @param xhat The encoded features
    *
    * */
  def i(xhat: I): I = reverseStack.forwardPass(xhat)

}

object GenericAutoEncoder {

  def apply[LayerP, I](
    learningAlgorithm: GradBasedBackPropagation[LayerP, I],
    initializer: RandomVariable[Seq[LayerP]]): GenericAutoEncoder[LayerP, I] =
    new GenericAutoEncoder(learningAlgorithm, initializer)

  def apply(neurons_by_layer: List[Int], acts: List[Activation[DenseVector[Double]]])
  : GenericAutoEncoder[(DenseMatrix[Double], DenseVector[Double]), DenseVector[Double]] = {

    require(
      neurons_by_layer.head == neurons_by_layer.last,
      "First and last layer of an auto-encoder must have the same dimensions!")

    val trainingAlg = new FFBackProp(NeuralStackFactory(neurons_by_layer)(acts))
    apply(trainingAlg, GenericFFNeuralNet.getWeightInitializer(neurons_by_layer))
  }

  def apply(inDim: Int, outDim: Int, acts: List[Activation[DenseVector[Double]]])
  : GenericAutoEncoder[(DenseMatrix[Double], DenseVector[Double]), DenseVector[Double]] =
    apply(List(inDim, outDim, inDim), acts)
  
}