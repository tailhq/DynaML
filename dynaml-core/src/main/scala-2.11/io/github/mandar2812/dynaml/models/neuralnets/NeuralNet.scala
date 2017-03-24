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

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Uniform
import io.github.mandar2812.dynaml.graph.NeuralGraph
import io.github.mandar2812.dynaml.models.ParameterizedLearner
import io.github.mandar2812.dynaml.optimization.GradBasedBackPropagation
import io.github.mandar2812.dynaml.pipes.DataPipe
import io.github.mandar2812.dynaml.probability.RandomVariable

/**
  * @author mandar2812 date 22/03/2017.
  *
  * Base member of the Neural Network API.
  * */
trait NeuralNet[
Data, BaseGraph, Input, Output,
Graph <: NeuralGraph[BaseGraph, Input, Output]] extends
  ParameterizedLearner[
    Data, Graph, Input, Output,
    Stream[(Input, Output)]] {

  val transform: DataPipe[Data, Stream[(Input, Output)]]

  val numPoints: Int


  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: Input) = params.forwardPass(point)

  def _neuralStack: Graph = params
}

/**
  * Base class for implementations of feed-forward neural network
  * models.
  *
  * @tparam Data The type of the training data.
  * @tparam LayerP The type of the layer parameters i.e. weights/connections etc.
  * @tparam I The type of the input features, output features and layer activations
  * */
class GenericFFNeuralNet[Data, LayerP, I](
  trainingAlgorithm: GradBasedBackPropagation[LayerP, I],
  data: Data, trans: DataPipe[Data, Stream[(I, I)]],
  layerInitializer: RandomVariable[Seq[LayerP]]) extends NeuralNet[
    Data, Seq[NeuralLayer[LayerP, I, I]],
    I, I, NeuralStack[LayerP, I]] {

  val stackFactory: NeuralStackFactory[LayerP, I] = trainingAlgorithm.stackFactory

  protected val generator: RandomVariable[Seq[LayerP]] = layerInitializer

  override protected val g: Data = data

  val num_layers: Int = stackFactory.layerFactories.length + 1

  val num_hidden_layers: Int = stackFactory.layerFactories.length - 1

  val activations: Seq[Activation[I]] = stackFactory.layerFactories.map(_.activationFunc)

  override val transform = trans

  override val numPoints = transform(g).length

  override protected var params: NeuralStack[LayerP, I] = initParams()

  override protected val optimizer: GradBasedBackPropagation[LayerP, I] = trainingAlgorithm

  override def initParams() = (generator.sample > stackFactory).run()

  /**
    * Learn the parameters
    * of the model which
    * are in a node of the
    * graph.
    *
    **/
  override def learn() = {
    params = optimizer.optimize(numPoints, transform(g), initParams())
  }


}

object GenericFFNeuralNet {
  /**
    * Create a feed forward neural net
    * @param trainingAlgorithm The optimization/training routine
    *                          as a [[GradBasedBackPropagation]] instance
    * @param data The training data
    * @param trans A data pipeline transforming the training data from
    *              type [[Data]] to [[Stream]] of input patterns and targets
    * @param layerInitializer A [[RandomVariable]] which generates samples for
    *                         the layer parameters.
    * */
  def apply[Data, LayerP, I](
    trainingAlgorithm: GradBasedBackPropagation[LayerP, I],
    data: Data, trans: DataPipe[Data, Stream[(I, I)]],
    layerInitializer: RandomVariable[Seq[LayerP]]) =
    new GenericFFNeuralNet[Data, LayerP, I](trainingAlgorithm, data, trans, layerInitializer)

  /**
    * Returns a random variable which enables sampling
    * of layer connection matrices, in the case of feed forward
    * neural networks operating on breeze vectors.
    *
    * */
  def getWeightInitializer(num_units_by_layer: Seq[Int])
  : RandomVariable[Seq[(DenseMatrix[Double], DenseVector[Double])]] = {

    val uni = new Uniform(-1.0, 1.0)

    RandomVariable(
      num_units_by_layer.sliding(2)
        .toSeq
        .map(l => (l.head, l.last))
        .map((c) => RandomVariable(() => (
          DenseMatrix.tabulate(c._2, c._1)((_, _) => uni.draw()),
          DenseVector.tabulate(c._2)(_ => uni.draw())))
        ):_*
    )
  }
}