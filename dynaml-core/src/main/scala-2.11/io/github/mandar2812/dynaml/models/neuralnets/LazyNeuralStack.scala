package io.github.mandar2812.dynaml.models.neuralnets


/**
  * @author mandar2812 date: 17/04/2017.
  *
  * A computation stack whose layers are lazily computed
  * */
class LazyNeuralStack[P, I](elements: Stream[NeuralLayer[P, I, I]]) extends
  GenericNeuralStack[P, I, Stream[NeuralLayer[P, I, I]]](elements) {

  self =>

  /**
    * Slice the stack according to a range.
    **/
  override def apply(r: Range) = new LazyNeuralStack[P, I](g.slice(r.min, r.max + 1))

  /**
    * Append another computation stack to the end of the
    * current one.
    **/
  override def ++[G <: Traversable[NeuralLayer[P, I, I]]](
    otherStack: GenericNeuralStack[P, I, G]) = new LazyNeuralStack[P, I](self.g ++ otherStack._layers)

  /**
    * Append a single computation layer to the stack.
    **/
  override def :+(computationLayer: NeuralLayer[P, I, I]) = new LazyNeuralStack[P, I](self.g :+ computationLayer)
}

object LazyNeuralStack {

  def apply[P, I](
    layerFunc: (Int) => NeuralLayer[P, I, I],
    num_layers: Int) = new LazyNeuralStack[P, I](
    (0 until num_layers).toStream.map(i => layerFunc(i))
  )
}