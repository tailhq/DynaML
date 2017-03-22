package io.github.mandar2812.dynaml.models.neuralnets

import io.github.mandar2812.dynaml.graph.NeuralGraph
import io.github.mandar2812.dynaml.models.ParameterizedLearner
import io.github.mandar2812.dynaml.pipes.{DataPipe, MetaPipe}

/**
  * Created by mandar on 22/03/2017.
  */
abstract class NeuralNet[
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
  * The basic building block of a neural computation stack.
  *
  * @tparam P The type of the parameters/connections of the layer.
  * @tparam I The type of the input supplied to the layer
  * @tparam J The type of the output.
  * */
trait NeuralLayer[P, I, J] {

  /**
    * The layer synapses or connection weights
    * */
  val parameters: P

  val localField: DataPipe[I, J]

  /**
    * Activation function
    * */
  val activationFunc: Activation[J]

  /**
    * Compute the forward pass through the layer.
    * */
  val forward: DataPipe[I, J] = localField > activationFunc

}

object NeuralLayer {

  /**
    * Create a neural computation layer i.e. [[NeuralLayer]] instance
    *
    * @tparam P The type of the layer parameters/weights/connections
    * @tparam I The type of input accepted by the computation layer
    * @param compute Represents the actual computation as a [[MetaPipe]]
    *                i.e. a data pipeline which takes as input a parameter value
    *                and outputs a pipe which represents the layer computation
    * @param activation The activation function
    * */
  def apply[P, I, J](compute: MetaPipe[P, I, J], activation: Activation[J])(params: P) =
    new NeuralLayer[P, I, J] {
      override val parameters = params
      override val activationFunc = activation
      override val localField = compute(parameters)
    }

}

/**
  * A network, represented as a stack of [[NeuralLayer]] objects.
  * */
class NeuralStack[P, I](elements: NeuralLayer[P, I, I]*)
  extends NeuralGraph[Seq[NeuralLayer[P, I, I]], I, I] {

  override protected val g: Seq[NeuralLayer[P, I, I]] = elements

  val layerParameters: Seq[P] = g.map(_.parameters)

  def _layers = g

  /**
    * Do a forward pass through the network outputting all the intermediate.
    * layer activations.
    * */
  def forwardPropagate(x: I): Seq[I] = g.scanLeft(x)((h, layer) => layer.forward(h))

  /**
    * Do a forward pass through the network outputting only the output layer activations.
    * */
  val forwardPass: (I) => I = (x: I) => g.foldLeft(x)((h, layer) => layer.forward(h))

  /**
    * Batch version of [[forwardPropagate()]]
    * */
  def forwardPropagateBatch[T <: Traversable[I]](d: T): Seq[T] = g.scanLeft(d)((h, layer) => layer.forward(h))

  /**
    * Batch version of [[forwardPass()]]
    * */
  def forwardPassBatch[T <: Traversable[I]](d: T): T = g.foldLeft(d)((h, layer) => layer.forward(h))

  /**
    * Slice the stack according to a range.
    * */
  def apply(r: Range): NeuralStack[P, I] = NeuralStack(g.slice(r.min, r.max + 1):_*)

  /**
    * Append another computation stack to the end of the
    * current one.
    * */
  def ++(otherStack: NeuralStack[P, I]): NeuralStack[P, I] = NeuralStack(this.g ++ otherStack.g :_*)

  /**
    * Append a single computation layer to the stack.
    * */
  def :+(computationLayer: NeuralLayer[P, I, I]): NeuralStack[P, I] = NeuralStack(this.g :+ computationLayer :_*)

}

object NeuralStack {

  def apply[P, I](elements: NeuralLayer[P, I, I]*): NeuralStack[P, I] = new NeuralStack(elements:_*)
}

/**
  * A mechanism to generate neural computation layers on the fly.
  * */
class NeuralLayerFactory[P, I, J](
  metaLocalField: MetaPipe[P, I, J],
  activationFunc: Activation[J]) extends
  DataPipe[P, NeuralLayer[P, I, J]] {

  override def run(params: P) = NeuralLayer(metaLocalField, activationFunc)(params)
}

class NeuralStackFactory[P, I](layerFacs: NeuralLayerFactory[P, I, I]*)
  extends DataPipe[Seq[P], NeuralStack[P, I]] {

  val layerFactories: Seq[NeuralLayerFactory[P, I, I]] = layerFacs

  override def run(params: Seq[P]): NeuralStack[P, I] = NeuralStack(layerFactories.zip(params).map(couple => {
    couple._1.run(couple._2)
  }):_*)
}

