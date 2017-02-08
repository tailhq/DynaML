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
  override def evaluateAt(config: Map[String, Double])(x: Index, y: Index) = {
    println("evaluateAt function not defined, continuing with a stub, "+
      "expect problem with kernel calculation!")
    0.0
  }

  override def gradientAt(config: Map[String, Double])(x: Index, y: Index) = {
    println("gradientAt function not defined, continuing with a stub ")

    effective_hyper_parameters.map((_, 0.0)).toMap
  }

}
