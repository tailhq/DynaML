package io.github.mandar2812.dynaml.models.neuralnets

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.graph.NeuralGraph
import io.github.mandar2812.dynaml.pipes.DataPipe

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

class NeuralStackFactory[P, I](layerFacs: NeuralLayerFactory[P, I, I]*)
  extends DataPipe[Seq[P], NeuralStack[P, I]] {

  val layerFactories: Seq[NeuralLayerFactory[P, I, I]] = layerFacs

  override def run(params: Seq[P]): NeuralStack[P, I] = NeuralStack(layerFactories.zip(params).map(couple => {
    couple._1.run(couple._2)
  }):_*)
}

object NeuralStackFactory {

  def apply[P, I](layerFacs: NeuralLayerFactory[P, I, I]*): NeuralStackFactory[P, I] =
    new NeuralStackFactory(layerFacs:_*)

  def apply(
    num_units_by_layer: Seq[Int])(
    activations: Seq[Activation[DenseVector[Double]]])
  : NeuralStackFactory[(DenseMatrix[Double], DenseVector[Double]), DenseVector[Double]] = {

    val layerFactories = num_units_by_layer.sliding(2).toSeq
      .map(l => (l.head, l.last)).zip(activations)
      .map((couple: ((Int, Int), Activation[DenseVector[Double]])) => {
        val ((inDim, outDim), actFunc) = couple
        new Vec2VecLayerFactory(actFunc)(inDim, outDim)
      })

    new NeuralStackFactory(layerFactories:_*)
  }
}

