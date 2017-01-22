package io.github.mandar2812.dynaml.kernels

import scalaxy.streams.optimize
import scala.reflect.ClassTag
import breeze.linalg.DenseMatrix
import io.github.mandar2812.dynaml.algebra.PartitionedPSDMatrix
import io.github.mandar2812.dynaml.pipes._

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
  def +[T <: LocalScalarKernel[Index]](otherKernel: T)(implicit ev: ClassTag[Index]): CompositeCovariance[Index] =
    new AdditiveCovariance[Index](this, otherKernel)

  /**
    *  Create composite kernel k = k<sub>1</sub> * k<sub>2</sub>
    *
    *  @param otherKernel The kernel to add to the current one.
    *  @return The kernel k defined above.
    *
    * */
  def *[T <: LocalScalarKernel[Index]](otherKernel: T)(implicit ev: ClassTag[Index]): CompositeCovariance[Index] =
    new MultiplicativeCovariance[Index](this, otherKernel)

  def :*[T1](otherKernel: LocalScalarKernel[T1]): CompositeCovariance[(Index, T1)] =
    new TensorCombinationKernel[Index, T1](this, otherKernel)
    //new KernelOps.PairOps[Index, T1].tensorMultLocalScKernels(this, otherKernel)

  def :+[T1](otherKernel: LocalScalarKernel[T1]): CompositeCovariance[(Index, T1)] =
    new TensorCombinationKernel[Index, T1](this, otherKernel)(Reducer.:+:)
    //new KernelOps.PairOps[Index, T1].tensorAddLocalScKernels(this, otherKernel)

  override def buildKernelMatrix[S <: Seq[Index]](mappedData: S, length: Int): KernelMatrix[DenseMatrix[Double]] =
    SVMKernel.buildSVMKernelMatrix[S, Index](mappedData, length, this.evaluate)

  override def buildCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
    SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

  def buildBlockedKernelMatrix[S <: Seq[Index]](mappedData: S, length: Long): PartitionedPSDMatrix =
    SVMKernel.buildPartitionedKernelMatrix(mappedData, length, rowBlocking, colBlocking, this.evaluate)

  def buildBlockedCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
    SVMKernel.crossPartitonedKernelMatrix(dataset1, dataset2, rowBlocking, colBlocking, this.evaluate)

}

abstract class CompositeCovariance[T]
  extends LocalSVMKernel[T] {
  override def repr: CompositeCovariance[T] = this
}







