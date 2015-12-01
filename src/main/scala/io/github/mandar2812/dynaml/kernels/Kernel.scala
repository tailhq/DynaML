package io.github.mandar2812.dynaml.kernels

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

  def setHyperParameters(h: Map[String, Double]): this.type = this

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
trait ScalarKernel[Index, Matrix] extends CovarianceFunction[Index, Double, Matrix] {
  def +(otherKernel: ScalarKernel[Index, Matrix]): CompositeCovariance[Index, Double, Matrix] = {

    val firstKernelHyp = this.hyper_parameters

    val firstKern = this.evaluate _

    val buildMat = this.buildKernelMatrix _
    val buildCrossMat = this.buildCrossKernelMatrix _

    new CompositeCovariance[Index, Double, Matrix] {
      override val hyper_parameters = firstKernelHyp ++ otherKernel.hyper_parameters

      override def evaluate(x: Index, y: Index) = firstKern(x,y) + otherKernel.evaluate(x,y)

      override def buildKernelMatrix[S <: Seq[Index]](mappedData: S, length: Int) =
        buildMat(mappedData, length)

      override def buildCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
        buildCrossMat(dataset1, dataset2)

    }
  }

  def *(otherKernel: ScalarKernel[Index, Matrix]): CompositeCovariance[Index, Double, Matrix] = {

    val firstKernelHyp = this.hyper_parameters

    val firstKern = this.evaluate _

    val buildMat = this.buildKernelMatrix _
    val buildCrossMat = this.buildCrossKernelMatrix _

    new CompositeCovariance[Index, Double, Matrix] {
      override val hyper_parameters = firstKernelHyp ++ otherKernel.hyper_parameters

      override def evaluate(x: Index, y: Index) = firstKern(x,y) * otherKernel.evaluate(x,y)

      override def buildKernelMatrix[S <: Seq[Index]](mappedData: S, length: Int) =
        buildMat(mappedData, length)

      override def buildCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
        buildCrossMat(dataset1, dataset2)

    }
  }

}


