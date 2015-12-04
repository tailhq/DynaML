package io.github.mandar2812.dynaml.kernels

import breeze.linalg.DenseMatrix


/**
 * Defines a base class for kernels
 * defined on arbitrary objects.
 *
 * @tparam T The domain over which kernel function
 *           k(x, y) is defined. i.e. x,y belong to T
 * @tparam V The type of value returned by the kernel function
 *           k(x,y)
 * */
trait Kernel[T, V] {
  def evaluate(x: T, y: T): V
}

/**
  * A covariance function implementation. Covariance functions are
  * central to Stochastic Process Models as well as SVMs.
  * */
abstract class CovarianceFunction[T, V, M] extends Kernel[T, V] {

  val hyper_parameters: List[String]

  var state: Map[String, Double] = Map()//hyper_parameters.map(key => (key, 1.0)).toMap

  def setHyperParameters(h: Map[String, Double]): this.type = {
    assert(hyper_parameters.forall(h contains _),
      "All hyper parameters must be contained in the arguments")
    hyper_parameters.foreach((key) => {
      state += (key -> h(key))
    })
    this
  }

  def buildKernelMatrix[S <: Seq[T]](mappedData: S,
                                     length: Int): KernelMatrix[M]

  def buildCrossKernelMatrix[S <: Seq[T]](dataset1: S, dataset2: S): M
}

abstract class CompositeCovariance[T, V, M]
  extends CovarianceFunction[T, V, M] {

}

/**
  * Scalar Kernel defines algebraic behavior for kernels of the form
  * K: Index x Index -> Double, i.e. kernel functions whose output
  * is a scalar/double value. Generic behavior for these kernels
  * is given by the ability to add and multiply valid kernels to
  * create new valid scalar kernel functions.
  *
  * */
trait LocalScalarKernel[Index] extends CovarianceFunction[Index, Double, DenseMatrix[Double]] {
  def +(otherKernel: LocalScalarKernel[Index]): CompositeCovariance[Index, Double, DenseMatrix[Double]] = {

    val firstKernelHyp = this.hyper_parameters

    val firstKern = this.evaluate _

    new CompositeCovariance[Index, Double, DenseMatrix[Double]] {
      override val hyper_parameters = firstKernelHyp ++ otherKernel.hyper_parameters

      override def evaluate(x: Index, y: Index) = firstKern(x,y) + otherKernel.evaluate(x,y)

      override def buildKernelMatrix[S <: Seq[Index]](mappedData: S, length: Int) =
        SVMKernel.buildSVMKernelMatrix[S, Index](mappedData, length, this.evaluate)

      override def buildCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
        SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

    }
  }

  def *(otherKernel: LocalScalarKernel[Index]): CompositeCovariance[Index, Double, DenseMatrix[Double]] = {

    val firstKernelHyp = this.hyper_parameters

    val firstKern = this.evaluate _

    new CompositeCovariance[Index, Double, DenseMatrix[Double]] {
      override val hyper_parameters = firstKernelHyp ++ otherKernel.hyper_parameters

      override def evaluate(x: Index, y: Index) = firstKern(x,y) * otherKernel.evaluate(x,y)

      override def buildKernelMatrix[S <: Seq[Index]](mappedData: S, length: Int) =
        SVMKernel.buildSVMKernelMatrix[S, Index](mappedData, length, this.evaluate)

      override def buildCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
        SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

    }
  }

}


