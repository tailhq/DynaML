package io.github.mandar2812.dynaml.kernels

import breeze.linalg.DenseMatrix
import io.github.mandar2812.dynaml.algebra.{PartitionedPSDMatrix, PartitionedVector}

/**
  * Scalar Kernel defines algebraic behavior for kernels of the form
  * K: Index x Index -> Double, i.e. kernel functions whose output
  * is a scalar/double value. Generic behavior for these kernels
  * is given by the ability to add and multiply valid kernels to
  * create new valid scalar kernel functions.
  *
  * */
trait LocalScalarKernel[Index] extends
CovarianceFunction[Index, Double, DenseMatrix[Double]]
  with KernelOps[LocalScalarKernel[Index]] {

  override def repr: LocalScalarKernel[Index] = this

  implicit protected val kernelOps = new KernelOps.Ops[Index]

  var (rowBlocking, colBlocking): (Int, Int) = (1000, 1000)

  def setBlockSizes(s: (Int, Int)): Unit = {
    rowBlocking = s._1
    colBlocking = s._2
  }

  def gradient(x: Index, y: Index): Map[String, Double] = effective_hyper_parameters.map((_, 0.0)).toMap

  /**
    *  Create composite kernel k = k<sub>1</sub> + k<sub>2</sub>
    *
    *  param otherKernel The kernel to add to the current one.
    *  return The kernel k defined above.
    *
    * */
  def +[T <: LocalScalarKernel[Index]](otherKernel: T): CompositeCovariance[Index] = this.+(otherKernel)

  /**
    *  Create composite kernel k = k<sub>1</sub> * k<sub>2</sub>
    *
    *  @param otherKernel The kernel to add to the current one.
    *  @return The kernel k defined above.
    *
    * */
  def *[T <: LocalScalarKernel[Index]](otherKernel: T): CompositeCovariance[Index] = this.*(otherKernel)

  def :*[T1](otherKernel: LocalScalarKernel[T1]): CompositeCovariance[(Index, T1)] =
    new KernelOps.PairOps[Index, T1].tensorMultLocalScKernels(this, otherKernel)

  def buildBlockedKernelMatrix[S <: Seq[Index]](mappedData: S, length: Long): PartitionedPSDMatrix =
    SVMKernel.buildPartitionedKernelMatrix(mappedData, length, rowBlocking, colBlocking, this.evaluate)

  def buildBlockedCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
    SVMKernel.crossPartitonedKernelMatrix(dataset1, dataset2, rowBlocking, colBlocking, this.evaluate)

}

abstract class CompositeCovariance[T]
  extends LocalSVMKernel[T] {
  override def repr: CompositeCovariance[T] = this
}

/**
  * A kernel/covariance function which can be seen as a combination
  * of base kernels over a subset of the input space.
  *
  * for example K((x1, y1), (x1, y2)) = k1(x1,x2) + k2(y1,y2)
  */
trait DecomposableCovariance extends CompositeCovariance[PartitionedVector]