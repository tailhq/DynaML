package io.github.mandar2812.dynaml.kernels

import breeze.generic.UFunc

/**
  * @author mandar2812 date: 26/10/2016.
  */
trait KernelOps[+This] extends Any {
  def repr: This

  /** Alias for :+(b) for all b. */
  final def +[TT >: This, B, That](b: B)(implicit op: KernelOpAdd.Impl2[TT, B, That]) = op(repr, b)

  final def *[TT >: This, B, That](b: B)(implicit op: KernelOpMult.Impl2[TT, B, That]) = op(repr, b)

  final def :*[TT >: This, B, That](b: B)(implicit op: KernelOuterMult.Impl2[TT, B, That]) = op(repr, b)

  final def :+[TT >: This, B, That](b: B)(implicit op: KernelOuterAdd.Impl2[TT, B, That]) = op(repr, b)

}

object KernelOps extends UFunc {

  class Ops[Index] extends Serializable {
    implicit object addLocalScKernels extends KernelOpAdd.Impl2[
        LocalScalarKernel[Index],
        LocalScalarKernel[Index],
        CompositeCovariance[Index]] {
      override def apply(
        firstKern: LocalScalarKernel[Index],
        otherKernel: LocalScalarKernel[Index]): CompositeCovariance[Index] =
        new CompositeCovariance[Index] {

          val (fID, sID) = (firstKern.toString.split("\\.").last, otherKernel.toString.split("\\.").last)

          override val hyper_parameters =
            firstKern.hyper_parameters.map(h => fID+"/"+h) ++
              otherKernel.hyper_parameters.map(h => sID+"/"+h)

          override def evaluate(x: Index, y: Index) = firstKern.evaluate(x,y) + otherKernel.evaluate(x,y)

          state = firstKern.state.map(h => (fID+"/"+h._1, h._2)) ++ otherKernel.state.map(h => (sID+"/"+h._1, h._2))

          blocked_hyper_parameters =
            firstKern.blocked_hyper_parameters.map(h => fID+"/"+h) ++
              otherKernel.blocked_hyper_parameters.map(h => sID+"/"+h)

          override def setHyperParameters(h: Map[String, Double]): this.type = {
            firstKern.setHyperParameters(h.filter(_._1.contains(fID))
              .map(kv => (kv._1.split("/").tail.mkString("/"), kv._2)))
            otherKernel.setHyperParameters(h.filter(_._1.contains(sID))
              .map(kv => (kv._1.split("/").tail.mkString("/"), kv._2)))
            this
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

          val (fID, sID) = (firstKern.toString.split("\\.").last, otherKernel.toString.split("\\.").last)

          override val hyper_parameters =
            firstKern.hyper_parameters.map(h => fID+"/"+h) ++
              otherKernel.hyper_parameters.map(h => sID+"/"+h)

          override def evaluate(x: Index, y: Index) = firstKern.evaluate(x,y) * otherKernel.evaluate(x,y)

          state = firstKern.state.map(h => (fID+"/"+h._1, h._2)) ++ otherKernel.state.map(h => (sID+"/"+h._1, h._2))

          blocked_hyper_parameters =
            firstKern.blocked_hyper_parameters.map(h => fID+"/"+h) ++
              otherKernel.blocked_hyper_parameters.map(h => sID+"/"+h)

          override def setHyperParameters(h: Map[String, Double]): this.type = {
            firstKern.setHyperParameters(h.filter(_._1.contains(fID))
              .map(kv => (kv._1.split("/").tail.mkString("/"), kv._2)))
            otherKernel.setHyperParameters(h.filter(_._1.contains(sID))
              .map(kv => (kv._1.split("/").tail.mkString("/"), kv._2)))
            this
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
  }

  class PairOps[Index, Index1] extends Serializable {
    implicit object tensorMultLocalScKernels
      extends KernelOuterMult.Impl2[
        LocalScalarKernel[Index],
        LocalScalarKernel[Index1],
        CompositeCovariance[(Index, Index1)]] {
      override def apply(firstkernel: LocalScalarKernel[Index],
                         otherKernel: LocalScalarKernel[Index1]): CompositeCovariance[(Index, Index1)] =

        new CompositeCovariance[(Index, Index1)] {

          override val hyper_parameters: List[String] = firstkernel.hyper_parameters ++ otherKernel.hyper_parameters

          state = firstkernel.state ++ otherKernel.state

          blocked_hyper_parameters = otherKernel.blocked_hyper_parameters ++ firstkernel.blocked_hyper_parameters

          override def setHyperParameters(h: Map[String, Double]): this.type = {
            firstkernel.setHyperParameters(h)
            otherKernel.setHyperParameters(h)
            super.setHyperParameters(h)
          }

          override def gradient(x: (Index, Index1), y: (Index, Index1)): Map[String, Double] =
            firstkernel.gradient(x._1, y._1).mapValues(v => v*otherKernel.evaluate(x._2, y._2)) ++
              otherKernel.gradient(x._2, y._2).mapValues(v => v*firstkernel.evaluate(x._1, y._1))

          override def buildKernelMatrix[S <: Seq[(Index, Index1)]](mappedData: S, length: Int) =
            SVMKernel.buildSVMKernelMatrix(mappedData, length, this.evaluate)

          override def buildCrossKernelMatrix[S <: Seq[(Index, Index1)]](dataset1: S, dataset2: S) =
            SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

          override def evaluate(x: (Index, Index1), y: (Index, Index1)): Double =
            firstkernel.evaluate(x._1, y._1)*otherKernel.evaluate(x._2, y._2)
        }
    }

    implicit object tensorAddLocalScKernels
      extends KernelOuterMult.Impl2[
        LocalScalarKernel[Index],
        LocalScalarKernel[Index1],
        CompositeCovariance[(Index, Index1)]] {
      override def apply(firstkernel: LocalScalarKernel[Index],
                         otherKernel: LocalScalarKernel[Index1]): CompositeCovariance[(Index, Index1)] =

        new CompositeCovariance[(Index, Index1)] {

          override val hyper_parameters: List[String] = firstkernel.hyper_parameters ++ otherKernel.hyper_parameters

          state = firstkernel.state ++ otherKernel.state

          blocked_hyper_parameters = otherKernel.blocked_hyper_parameters ++ firstkernel.blocked_hyper_parameters

          override def setHyperParameters(h: Map[String, Double]): this.type = {
            firstkernel.setHyperParameters(h)
            otherKernel.setHyperParameters(h)
            super.setHyperParameters(h)
          }

          override def gradient(x: (Index, Index1), y: (Index, Index1)): Map[String, Double] =
            firstkernel.gradient(x._1, y._1) ++ otherKernel.gradient(x._2,y._2)

          override def buildKernelMatrix[S <: Seq[(Index, Index1)]](mappedData: S, length: Int) =
            SVMKernel.buildSVMKernelMatrix(mappedData, length, this.evaluate)

          override def buildCrossKernelMatrix[S <: Seq[(Index, Index1)]](dataset1: S, dataset2: S) =
            SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

          override def evaluate(x: (Index, Index1), y: (Index, Index1)): Double =
            firstkernel.evaluate(x._1, y._1)+otherKernel.evaluate(x._2, y._2)
        }
    }

    implicit object tensorAddPartLocalScKernels
      extends KernelOuterMult.Impl2[
        LocalSVMKernel[Index],
        LocalSVMKernel[Index],
        CompositeCovariance[(Index, Index)]] {
      override def apply(firstkernel: LocalSVMKernel[Index],
                         otherKernel: LocalSVMKernel[Index]): CompositeCovariance[(Index, Index)] =

        new CompositeCovariance[(Index, Index)] {

          override val hyper_parameters: List[String] = firstkernel.hyper_parameters ++ otherKernel.hyper_parameters

          state = firstkernel.state ++ otherKernel.state

          blocked_hyper_parameters = otherKernel.blocked_hyper_parameters ++ firstkernel.blocked_hyper_parameters

          override def setHyperParameters(h: Map[String, Double]): this.type = {
            firstkernel.setHyperParameters(h)
            otherKernel.setHyperParameters(h)
            super.setHyperParameters(h)
          }

          override def gradient(x: (Index, Index), y: (Index, Index)): Map[String, Double] =
            firstkernel.gradient(x._1, y._1) ++ otherKernel.gradient(x._2,y._2)

          override def buildKernelMatrix[S <: Seq[(Index, Index)]](mappedData: S, length: Int) =
            SVMKernel.buildSVMKernelMatrix(mappedData, length, this.evaluate)

          override def buildCrossKernelMatrix[S <: Seq[(Index, Index)]](dataset1: S, dataset2: S) =
            SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

          override def evaluate(x: (Index, Index), y: (Index, Index)): Double =
            firstkernel.evaluate(x._1, y._1)+otherKernel.evaluate(x._2, y._2)
        }
    }



  }

}