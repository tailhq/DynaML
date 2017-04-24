package io.github.mandar2812.dynaml.models.neuralnets

import io.github.mandar2812.dynaml.graph.NeuralGraph

import scala.collection.GenTraversableLike

/**
  * @author mandar2812 date 17/04/2017.
  *
  * Base class for Neural Computation Stack
  * implementations.
  * */
class GenericNeuralStack[
P, I, Layer <: NeuralLayer[P, I, I],
T[L] <: Traversable[L] with GenTraversableLike[L, T[L]]
](elements: T[Layer]) extends NeuralGraph[T[Layer], I, I] {

  self =>

  override protected val g: T[Layer] = elements

  /**
    * Do a forward pass through the network outputting only the output layer activations.
    * */
  override val forwardPass: (I) => I = (x: I) => g.foldLeft(x)((h, layer) => layer.forward(h))

  def _layers = g

  /**
    * Do a forward pass through the network outputting all the intermediate.
    * layer activations.
    * */
  def forwardPropagate(x: I): T[I] = g.scanLeft(x)((h, layer) => layer.forward(h)).asInstanceOf[T[I]]

  /**
    * Batch version of [[forwardPropagate()]]
    * */
  def forwardPropagateBatch[G <: Traversable[I]](d: G): T[G] =
    g.scanLeft(d)((h, layer) => layer.forward(h)).asInstanceOf[T[G]]

  /**
    * Batch version of [[forwardPass()]]
    * */
  def forwardPassBatch[G <: Traversable[I]](d: G): G = g.foldLeft(d)((h, layer) => layer.forward(h))

  /**
    * Slice the stack according to a range.
    * */
  def apply(r: Range): GenericNeuralStack[P, I, Layer, T] =
    new GenericNeuralStack(self.g.slice(r.min,r.max+1).asInstanceOf[T[Layer]])

  /**
    * Append another computation stack to the end of the
    * current one.
    * */
  def ++[
  L <: NeuralLayer[P, I, I],
  G[L1] <: Traversable[L1] with GenTraversableLike[L1, G[L1]]](
    otherStack: GenericNeuralStack[P, I, L, G])
  : GenericNeuralStack[P, I, NeuralLayer[P, I, I], T] = new GenericNeuralStack[P, I, NeuralLayer[P, I, I], T](
    (self.g.map((l: Layer) => l.asInstanceOf[NeuralLayer[P, I, I]]) ++
      otherStack._layers.map((l: L) => l.asInstanceOf[NeuralLayer[P, I, I]]))
      .asInstanceOf[T[NeuralLayer[P, I, I]]])

  /**
    * Append a single computation layer to the stack.
    * */
  def :+(computationLayer: NeuralLayer[P, I, I])
  : GenericNeuralStack[P, I, NeuralLayer[P, I, I], T] = self ++
    new GenericNeuralStack[P, I, NeuralLayer[P, I, I], Seq](Seq(computationLayer))

}
