package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{DenseMatrix, DenseVector}

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

  var blocked_hyper_parameters: List[String] = List()

  var state: Map[String, Double] = Map()

  def effective_state:Map[String, Double] =
    state.filterNot(h => blocked_hyper_parameters.contains(h._1))

  def effective_hyper_parameters: List[String] =
    hyper_parameters.filterNot(h => blocked_hyper_parameters.contains(h))

  def setHyperParameters(h: Map[String, Double]): this.type = {
    assert(effective_hyper_parameters.forall(h.contains),
      "All hyper parameters must be contained in the arguments")
    effective_hyper_parameters.foreach((key) => {
      state += (key -> h(key))
    })
    this
  }

  def gradient(x: T, y: T): Map[String, V]

  def buildKernelMatrix[S <: Seq[T]](mappedData: S,
                                     length: Int): KernelMatrix[M]

  def buildCrossKernelMatrix[S <: Seq[T]](dataset1: S, dataset2: S): M

}

object CovarianceFunction {
  def apply[T](phi: T => DenseVector[Double]) = new LocalScalarKernel[T] {
    override val hyper_parameters: List[String] = List()

    override def evaluate(x: T, y: T): Double = phi(x) dot phi(y)

    override def buildKernelMatrix[S <: Seq[T]](mappedData: S, length: Int): KernelMatrix[DenseMatrix[Double]] =
      SVMKernel.buildSVMKernelMatrix(mappedData, length, this.evaluate)

    override def buildCrossKernelMatrix[S <: Seq[T]](dataset1: S, dataset2: S): DenseMatrix[Double] =
      SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

  }

  def apply[T](phi: Map[String, Double] => T => DenseVector[Double])(s: Map[String, Double]) =
    new LocalScalarKernel[T] {
      override val hyper_parameters: List[String] = s.keys.toList

      state = s

      override def evaluate(x: T, y: T): Double = phi(state)(x) dot phi(state)(y)

      override def buildKernelMatrix[S <: Seq[T]](mappedData: S, length: Int): KernelMatrix[DenseMatrix[Double]] =
        SVMKernel.buildSVMKernelMatrix(mappedData, length, this.evaluate)

      override def buildCrossKernelMatrix[S <: Seq[T]](dataset1: S, dataset2: S): DenseMatrix[Double] =
        SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

    }
}

abstract class CompositeCovariance[T]
  extends LocalScalarKernel[T] {

}

/**
  * Scalar Kernel defines algebraic behavior for kernels of the form
  * K: Index x Index -> Double, i.e. kernel functions whose output
  * is a scalar/double value. Generic behavior for these kernels
  * is given by the ability to add and multiply valid kernels to
  * create new valid scalar kernel functions.
  *
  * */
trait LocalScalarKernel[Index] extends
CovarianceFunction[Index, Double, DenseMatrix[Double]] {

  def gradient(x: Index, y: Index): Map[String, Double] = effective_hyper_parameters.map((_, 0.0)).toMap

  /**
    *  Create composite kernel k = k<sub>1</sub> + k<sub>2</sub>
    *
    *  @param otherKernel The kernel to add to the current one.
    *  @return The kernel k defined above.
    *
    * */
  def +[T <: LocalScalarKernel[Index]](otherKernel: T): CompositeCovariance[Index] = {

    val firstKern = this

    new CompositeCovariance[Index] {
      override val hyper_parameters = firstKern.hyper_parameters ++ otherKernel.hyper_parameters

      override def evaluate(x: Index, y: Index) = firstKern.evaluate(x,y) + otherKernel.evaluate(x,y)

      state = firstKern.state ++ otherKernel.state

      blocked_hyper_parameters = firstKern.blocked_hyper_parameters ++ otherKernel.blocked_hyper_parameters

      override def setHyperParameters(h: Map[String, Double]): this.type = {
        firstKern.setHyperParameters(h)
        otherKernel.setHyperParameters(h)
        super.setHyperParameters(h)
      }

      override def gradient(x: Index, y: Index): Map[String, Double] =
        firstKern.gradient(x, y) ++ otherKernel.gradient(x,y)

      override def buildKernelMatrix[S <: Seq[Index]](mappedData: S, length: Int) =
        SVMKernel.buildSVMKernelMatrix[S, Index](mappedData, length, this.evaluate)

      override def buildCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
        SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

    }
  }

  /**
    *  Create composite kernel k = k<sub>1</sub> * k<sub>2</sub>
    *
    *  @param otherKernel The kernel to add to the current one.
    *  @return The kernel k defined above.
    *
    * */
  def *[T <: LocalScalarKernel[Index]](otherKernel: T): CompositeCovariance[Index] = {

    val firstKern = this

    new CompositeCovariance[Index] {
      override val hyper_parameters = firstKern.hyper_parameters ++ otherKernel.hyper_parameters

      override def evaluate(x: Index, y: Index) = firstKern.evaluate(x,y) * otherKernel.evaluate(x,y)

      state = firstKern.state ++ otherKernel.state

      blocked_hyper_parameters = firstKern.blocked_hyper_parameters ++ otherKernel.blocked_hyper_parameters

      override def setHyperParameters(h: Map[String, Double]): this.type = {
        firstKern.setHyperParameters(h)
        otherKernel.setHyperParameters(h)
        super.setHyperParameters(h)
      }

      override def gradient(x: Index, y: Index): Map[String, Double] =
        firstKern.gradient(x, y).map((couple) => (couple._1, couple._2*otherKernel.evaluate(x,y))) ++
          otherKernel.gradient(x,y).map((couple) => (couple._1, couple._2*firstKern.evaluate(x,y)))

      override def buildKernelMatrix[S <: Seq[Index]](mappedData: S, length: Int) =
        SVMKernel.buildSVMKernelMatrix[S, Index](mappedData, length, this.evaluate)

      override def buildCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
        SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

    }
  }

  def :*[T1](otherKernel: LocalScalarKernel[T1]): CompositeCovariance[(Index, T1)] = {

    val firstkernel = this

    new CompositeCovariance[(Index, T1)] {

      override val hyper_parameters: List[String] = firstkernel.hyper_parameters ++ otherKernel.hyper_parameters

      state = firstkernel.state ++ otherKernel.state

      blocked_hyper_parameters = otherKernel.blocked_hyper_parameters ++ firstkernel.blocked_hyper_parameters

      override def setHyperParameters(h: Map[String, Double]): this.type = {
        firstkernel.setHyperParameters(h)
        otherKernel.setHyperParameters(h)
        super.setHyperParameters(h)
      }

      override def gradient(x: (Index, T1), y: (Index, T1)): Map[String, Double] =
        firstkernel.gradient(x._1, y._1).mapValues(v => v*otherKernel.evaluate(x._2, y._2)) ++
          otherKernel.gradient(x._2, y._2).mapValues(v => v*firstkernel.evaluate(x._1, y._1))

      override def buildKernelMatrix[S <: Seq[(Index, T1)]](mappedData: S, length: Int) =
        SVMKernel.buildSVMKernelMatrix(mappedData, length, this.evaluate)

      override def buildCrossKernelMatrix[S <: Seq[(Index, T1)]](dataset1: S, dataset2: S) =
        SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

      override def evaluate(x: (Index, T1), y: (Index, T1)): Double =
        firstkernel.evaluate(x._1, y._1)*otherKernel.evaluate(x._2, y._2)
    }

  }


}


