package io.github.mandar2812.dynaml.models.neuralnets

import scala.collection.GenTraversableLike
import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.pipes.DataPipe


/**
  * A network, represented as a stack of [[NeuralLayer]] objects.
  * */
class NeuralStack[P, I](elements: Seq[NeuralLayer[P, I, I]])
  extends GenericNeuralStack[P, I, NeuralLayer[P, I, I], Seq](elements) {

  self =>

  override protected val g: Seq[NeuralLayer[P, I, I]] = elements

  val layerParameters: Seq[P] = g.map(_.parameters)

  /**
    * Slice the stack according to a range.
    * */
  override def apply(r: Range): NeuralStack[P, I] = NeuralStack(g.slice(r.min, r.max + 1):_*)


  /**
    * Append another computation stack to the end of the
    * current one.
    **/
  override def ++[
  L <: NeuralLayer[P, I, I],
  G[L1] <: Traversable[L1] with GenTraversableLike[L1, G[L1]]](
    otherStack: GenericNeuralStack[P, I, L, G]) =
    new NeuralStack(self.g ++ otherStack._layers.asInstanceOf[Seq[NeuralLayer[P, I, I]]])

  /**
    * Append a single computation layer to the stack.
    * */
  override def :+(computationLayer: NeuralLayer[P, I, I]): NeuralStack[P, I] =
    NeuralStack(self.g :+ computationLayer :_*)

}

object NeuralStack {

  def apply[P, I](elements: NeuralLayer[P, I, I]*): NeuralStack[P, I] = new NeuralStack(elements)
}

/**
  * Creates a [[NeuralStack]] on the fly. Represented as a [[DataPipe]]
  *
  * @tparam P The type of layer parameters
  * @tparam I The type of input accepted by the computational stack generated
  *           by the [[run()]] method.
  *
  * @param layerFacs A collection of [[NeuralLayerFactory]] instances.
  * */
class NeuralStackFactory[P, I](layerFacs: NeuralLayerFactory[P, I, I]*)
  extends DataPipe[Seq[P], NeuralStack[P, I]] {

  val layerFactories: Seq[NeuralLayerFactory[P, I, I]] = layerFacs

  /**
    * Generate a [[NeuralStack]]
    * @param params The layer parameters/weights
    * @return A [[NeuralStack]] with the provided layer weights.
    * */
  override def run(params: Seq[P]): NeuralStack[P, I] = NeuralStack(
    layerFactories.zip(params).map(couple => couple._1.run(couple._2)):_*
  )
}

/**
  * Contains convenience methods for instantiating [[NeuralStack]] objects.
  * */
object NeuralStackFactory {

  /**
    * Create a [[NeuralStackFactory]] from a list of [[NeuralLayerFactory]] instances.
    * This is an alias for instance creation using the `new` keyword.
    * */
  def apply[P, I](layerFacs: NeuralLayerFactory[P, I, I]*): NeuralStackFactory[P, I] =
    new NeuralStackFactory(layerFacs:_*)

  /**
    * Create a [[NeuralStackFactory]] which
    * performs operations on breeze [[DenseVector]] instances.
    *
    * The parameters of each layer are represented
    * as a [[DenseMatrix]] (W) and a [[DenseVector]] (b)
    * yielding the following relation for the forward pass.
    *
    * a(n+1) = &sigma;(W.a(n) + b)
    *
    * @param num_units_by_layer A Sequence containing number of neurons for each layer
    * @param activations The activation functions for each layer, each activation is
    *                    an instance or sub-class instance of [[Activation]].
    *
    * @return A [[NeuralStackFactory]] instance which operates on breeze [[DenseVector]],
    *         which generates [[NeuralStack]] instances having the specified activations
    *         and provided layer parameters.
    * */
  def apply(
    num_units_by_layer: Seq[Int])(
    activations: Seq[Activation[DenseVector[Double]]])
  : NeuralStackFactory[(DenseMatrix[Double], DenseVector[Double]), DenseVector[Double]] = {

    require(
      num_units_by_layer.length == activations.length + 1,
      "Number of layers (input, output & hidden) must be = num(activation layers) + 1")

    val layerFactories = num_units_by_layer.sliding(2).toSeq
      .map(l => (l.head, l.last)).zip(activations)
      .map((couple: ((Int, Int), Activation[DenseVector[Double]])) => {
        val ((inDim, outDim), actFunc) = couple
        new Vec2VecLayerFactory(actFunc)(inDim, outDim)
      })

    new NeuralStackFactory(layerFactories:_*)
  }
}

