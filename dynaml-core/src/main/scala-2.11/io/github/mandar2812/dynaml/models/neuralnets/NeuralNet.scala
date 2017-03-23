package io.github.mandar2812.dynaml.models.neuralnets

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

  val numPoints = transform(g).length

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

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: Input) = params.forwardPass(point)
}

/**
  * Base class for implementations of feed-forward neural network
  * models.
  *
  * @tparam Data The type of the training data.
  * @tparam LayerP The type of the layer parameters i.e. weights/connections etc.
  * @tparam I The type of the input features, output features and layer activations
  * */
abstract class GenericFFNeuralNet[Data, LayerP, I]
  extends NeuralNet[
    Data, Seq[NeuralLayer[LayerP, I, I]],
    I, I, NeuralStack[LayerP, I]] {

  val stackFactory: NeuralStackFactory[LayerP, I]

  protected val generator: RandomVariable[LayerP]

  val num_layers: Int = stackFactory.layerFactories.length + 1

  val num_hidden_layers: Int = stackFactory.layerFactories.length - 1

  val activations: Seq[Activation[I]] = stackFactory.layerFactories.map(_.activationFunc)

  override def initParams() = stackFactory(generator.iid(activations.length).sample())

  override protected var params: NeuralStack[LayerP, I] = initParams()

  override protected val optimizer: GradBasedBackPropagation[LayerP, I]

}

object GenericFFNeuralNet {
  /**
    * Create a feed forward neural net
    * @param networkFactory A [[NeuralStackFactory]] object.
    * @param trainingAlgorithm The optimization/training routine
    *                          as a [[GradBasedBackPropagation]] instance
    * @param data The training data
    * @param trans A data pipeline transforming the training data from
    *              type [[Data]] to [[Stream]] of input patterns and targets
    * @param layerInitializer A [[RandomVariable]] which generates samples for
    *                         the layer parameters.
    * */
  def apply[Data, LayerP, I](
    networkFactory: NeuralStackFactory[LayerP, I],
    trainingAlgorithm: GradBasedBackPropagation[LayerP, I],
    data: Data, trans: DataPipe[Data, Stream[(I, I)]],
    layerInitializer: RandomVariable[LayerP]) =
    new GenericFFNeuralNet[Data, LayerP, I] {

      override val stackFactory = networkFactory
      override protected val generator: RandomVariable[LayerP] = layerInitializer
      override protected val optimizer: GradBasedBackPropagation[LayerP, I] = trainingAlgorithm
      override val transform = trans
      override protected val g: Data = data
  }
}