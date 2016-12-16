package io.github.mandar2812.dynaml.kernels

import breeze.linalg.DenseMatrix
import io.github.mandar2812.dynaml.algebra.PartitionedPSDMatrix

/**
  * Kernels with a locally stored matrix in the form
  * of a breeze [[DenseMatrix]] instance. Optionally
  * a kernel matrix stored as a [[PartitionedPSDMatrix]]
  * can also be generated.
  * */
trait LocalSVMKernel[Index] extends LocalScalarKernel[Index] {

  /*def :+(otherKernel: LocalSVMKernel[Index]): CompositeCovariance[(Index, Index)] =
    new KernelOps.PairOps[Index, Index].tensorAddPartLocalScKernels(this, otherKernel)

  def :*(otherKernel: LocalSVMKernel[Index]): CompositeCovariance[(Index, Index)] =
    new KernelOps.PairOps[Index, Index].tensorMultLocalScKernels(this, otherKernel)*/


}
