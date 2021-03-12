package io.github.tailhq.dynaml.kernels

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.tailhq.dynaml.pipes.DataPipe
import spire.algebra.InnerProductSpace

/**
  * A (symmeteric positive definite) covariance function. Covariance functions are
  * central to Gaussian/Student T Process Models as well as SVMs.
  *
  * @tparam T The index set over which K(.,.) is defined K: T &times; T -> V
  * @tparam V The value outputted by the kernel
  * @tparam M The type of the kernel matrix object.
  * @author tailhq
  *
  * */
abstract class CovarianceFunction[T, V, M] extends Kernel[T, V] with Serializable {

  val hyper_parameters: List[String]

  var blocked_hyper_parameters: List[String] = List()

  var state: Map[String, Double] = Map()

  def block(h: String*) = blocked_hyper_parameters = List(h:_*)

  def block_all_hyper_parameters: Unit = {
    blocked_hyper_parameters = hyper_parameters
  }

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

  def evaluateAt(config: Map[String, Double])(x: T, y: T): V

  def gradientAt(config: Map[String, Double])(x: T, y: T): Map[String, V]

  override def evaluate(x: T, y: T) = evaluateAt(state)(x, y)

  def gradient(x: T, y: T): Map[String, V] = gradientAt(state)(x, y)

  def buildKernelMatrix[S <: Seq[T]](mappedData: S,
                                     length: Int): KernelMatrix[M]

  def buildCrossKernelMatrix[S <: Seq[T]](dataset1: S, dataset2: S): M

}

object CovarianceFunction {

  /**
    * Create a kernel from a feature mapping.
    * K(x,y) = phi^T^(x) . phi(y)
    *
    * @param phi A general non linear transformation from the domain to
    *            a multidimensional vector.
    *
    * @return A kernel instance defined for that particular feature transformation.
    * */
  def apply[T](phi: T => DenseVector[Double])(
    implicit e: InnerProductSpace[DenseVector[Double], Double]) =
    new FeatureMapCovariance[T, DenseVector[Double]](DataPipe(phi)) {
      override val hyper_parameters: List[String] = List()

      override def evaluate(x: T, y: T): Double = phi(x) dot phi(y)

      override def buildKernelMatrix[S <: Seq[T]](mappedData: S, length: Int): KernelMatrix[DenseMatrix[Double]] =
        SVMKernel.buildSVMKernelMatrix(mappedData, length, this.evaluate)

      override def buildCrossKernelMatrix[S <: Seq[T]](dataset1: S, dataset2: S): DenseMatrix[Double] =
        SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

    }

  /**
    * Create a kernel from a feature mapping.
    * K(x,y) = phi^T^(x) . phi(y)
    *
    * @param phi A general non linear transformation as a [[DataPipe]]
    *            from the domain to a multidimensional vector.
    *
    * @return A kernel instance defined for that particular feature transformation.
    * */
  def apply[T](phi: DataPipe[T, DenseVector[Double]])(
    implicit e: InnerProductSpace[DenseVector[Double], Double]) =
    new FeatureMapCovariance[T, DenseVector[Double]](phi) {
      override val hyper_parameters: List[String] = List()

      override def evaluate(x: T, y: T): Double = phi(x) dot phi(y)

      override def buildKernelMatrix[S <: Seq[T]](mappedData: S, length: Int): KernelMatrix[DenseMatrix[Double]] =
        SVMKernel.buildSVMKernelMatrix(mappedData, length, this.evaluate)

      override def buildCrossKernelMatrix[S <: Seq[T]](dataset1: S, dataset2: S): DenseMatrix[Double] =
        SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

    }


  /**
    * Create a kernel from a symmetric function.
    *
    * K(x,y) = f(state)(x,y)
    *
    * @param phi  A function which for every state outputs a symmetric kernel
    *             evaluation function for inputs.
    * @param s The (beginning) state of the kernel.
    *
    * */
  def apply[T](phi: Map[String, Double] => (T, T) => Double)(s: Map[String, Double]) =
    new LocalSVMKernel[T] {
      override val hyper_parameters: List[String] = s.keys.toList

      state = s

      override def evaluateAt(config: Map[String, Double])(x: T, y: T): Double = phi(config)(x, y)

      override def buildKernelMatrix[S <: Seq[T]](mappedData: S, length: Int): KernelMatrix[DenseMatrix[Double]] =
        SVMKernel.buildSVMKernelMatrix(mappedData, length, evaluate)

      override def buildCrossKernelMatrix[S <: Seq[T]](dataset1: S, dataset2: S): DenseMatrix[Double] =
        SVMKernel.crossKernelMatrix(dataset1, dataset2, evaluate)

    }
}