package io.github.mandar2812.dynaml.kernels

import breeze.linalg.DenseMatrix
import io.github.mandar2812.dynaml.algebra.{KernelMatrix, PartitionedPSDMatrix}

/**
  * Kernels with a locally stored matrix in the form
  * of a breeze [[DenseMatrix]] instance. Optionally
  * a kernel matrix stored as a [[PartitionedPSDMatrix]]
  * can also be generated.
  * */
trait LocalSVMKernel[Index] extends LocalScalarKernel[Index] {

  var (rowBlocking, colBlocking): (Int, Int) = (1000, 1000)

  def setBlockSizes(s: (Int, Int)): Unit = {
    rowBlocking = s._1
    colBlocking = s._2
  }

  override def buildKernelMatrix[S <: Seq[Index]](
    mappedData: S,
    length: Int): KernelMatrix[DenseMatrix[Double]] =
    SVMKernel.buildSVMKernelMatrix[S, Index](mappedData, length, this.evaluate)

  override def buildCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
    SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

  def buildBlockedKernelMatrix[S <: Seq[Index]](mappedData: S, length: Long): PartitionedPSDMatrix =
    SVMKernel.buildPartitionedKernelMatrix(mappedData, length, rowBlocking, colBlocking, this.evaluate)

  def buildBlockedCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
    SVMKernel.crossPartitonedKernelMatrix(dataset1, dataset2, rowBlocking, colBlocking, this.evaluate)

}
