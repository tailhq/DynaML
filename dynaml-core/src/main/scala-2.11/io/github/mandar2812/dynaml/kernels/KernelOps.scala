package io.github.mandar2812.dynaml.kernels

import breeze.generic.UFunc

/**
  * Created by mandar on 26/10/2016.
  */
trait KernelOps[+This] extends Any {
  def repr: This

  /** Alias for :+(b) for all b. */
  final def +[TT >: This, B, That](b: B)(implicit op: KernelOpAdd.Impl2[TT, B, That]) = {
    op(repr, b)
  }

}

object KernelOps extends UFunc {

  class Ops[Index] {
    implicit object addLocalScKernels extends KernelOpAdd.Impl2[
        LocalScalarKernel[Index],
        LocalScalarKernel[Index],
        CompositeCovariance[Index]] {
      override def apply(
        firstKern: LocalScalarKernel[Index],
        otherKernel: LocalScalarKernel[Index]): CompositeCovariance[Index] =
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

    implicit object multLocalScKernels extends KernelOpMult.Impl2[
      LocalScalarKernel[Index],
      LocalScalarKernel[Index],
      CompositeCovariance[Index]] {
      override def apply(firstKern: LocalScalarKernel[Index],
                         otherKernel: LocalScalarKernel[Index]): CompositeCovariance[Index] =
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

    implicit object kroneckerProductKernels extends KernelKMult.Impl2

  }
}