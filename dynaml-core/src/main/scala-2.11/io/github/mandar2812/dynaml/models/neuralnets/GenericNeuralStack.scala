package io.github.mandar2812.dynaml.models.neuralnets

import io.github.mandar2812.dynaml.graph.NeuralGraph

/**
  * @author mandar2812 date 17/04/2017.
  *
  * Base class for Neural Computation Stack
  * implementations.
  * */
abstract class GenericNeuralStack[
P, I, T <: Traversable[NeuralLayer[P, I, I]]
](elements: T) extends NeuralGraph[T, I, I] {

  self =>

  override protected val g: T = elements

  /**
    * Do a forward pass through the network outputting only the output layer activations.
    * */
  override val forwardPass: (I) => I = (x: I) => g.foldLeft(x)((h, layer) => layer.forward(h))

  def _layers = g

  /**
    * Do a forward pass through the network outputting all the intermediate.
    * layer activations.
    * */
  def forwardPropagate(x: I): Traversable[I] = g.scanLeft(x)((h, layer) => layer.forward(h))

  /**
    * Batch version of [[forwardPropagate()]]
    * */
  def forwardPropagateBatch[G <: Traversable[I]](d: G): Traversable[G] = g.scanLeft(d)((h, layer) => layer.forward(h))

  /**
    * Batch version of [[forwardPass()]]
    * */
  def forwardPassBatch[G <: Traversable[I]](d: G): G = g.foldLeft(d)((h, layer) => layer.forward(h))

  /**
    * Slice the stack according to a range.
    * */
  def apply(r: Range): GenericNeuralStack[P, I, T]

  /**
    * Append another computation stack to the end of the
    * current one.
    * */
  def ++[G <: Traversable[NeuralLayer[P, I, I]]](otherStack: GenericNeuralStack[P, I, G]): GenericNeuralStack[P, I, T]

  /**
    * Append a single computation layer to the stack.
    * */
  def :+(computationLayer: NeuralLayer[P, I, I]): GenericNeuralStack[P, I, T]

}
