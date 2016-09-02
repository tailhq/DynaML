package io.github.mandar2812.dynaml.kernels

import breeze.linalg.DenseMatrix
import io.github.mandar2812.dynaml.analysis.KernelMatrix

/**
  * Kernels with a locally stored matrix in the form
  * of a breeze [[DenseMatrix]] instance.
  * */
trait LocalSVMKernel[Index] extends LocalScalarKernel[Index] {
  override def buildKernelMatrix[S <: Seq[Index]](
    mappedData: S,
    length: Int): KernelMatrix[DenseMatrix[Double]] =
    SVMKernel.buildSVMKernelMatrix[S, Index](mappedData, length, this.evaluate)

  override def buildCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
    SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)
}
