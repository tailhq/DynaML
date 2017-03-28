package io.github.mandar2812.dynaml.kernels

import scala.reflect.ClassTag
import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.algebra.PartitionedPSDMatrix
import io.github.mandar2812.dynaml.pipes._
import spire.algebra.InnerProductSpace

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
  with KernelOps[LocalScalarKernel[Index]] with Serializable { self =>

  override def repr: LocalScalarKernel[Index] = this

  implicit protected val kernelOps = new KernelOps.Ops[Index]

  var (rowBlocking, colBlocking): (Int, Int) = (1000, 1000)

  def setBlockSizes(s: (Int, Int)): Unit = {
    rowBlocking = s._1
    colBlocking = s._2
  }



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

  /**
    * Returns the kernel multiplied by a positive constant: k_new = k*c
    * */
  def *(c: Double): LocalScalarKernel[Index] = {
    require (c > 0, "Multiplicative constant applied on a kernel must be positive!")
    new LocalScalarKernel[Index] {

      override val hyper_parameters = self.hyper_parameters

      state = self.state

      blocked_hyper_parameters = self.blocked_hyper_parameters

      override def setHyperParameters(h: Map[String, Double]) = {
        self.setHyperParameters(h)
        super.setHyperParameters(h)
      }

      override def gradientAt(config: Map[String, Double])(x: Index, y: Index) =
        self.gradientAt(config)(x, y).map(co => (co._1, co._2*c))

      override def evaluateAt(config: Map[String, Double])(x: Index, y: Index) =
        self.evaluateAt(config)(x, y)*c

    }
  }

  def >[K <: GenericRBFKernel[Index]](otherKernel: K): CompositeCovariance[Index] = {

    new CompositeCovariance[Index] {

      override val hyper_parameters = self.hyper_parameters ++ otherKernel.hyper_parameters

      override def evaluateAt(config: Map[String, Double])(x: Index, y: Index) = {
        val arg = self.evaluateAt(config)(x,y) +
          self.evaluateAt(config)(y,y) -
          2.0*self.evaluateAt(config)(x,y)

        math.exp(-1.0*arg/(2.0*math.pow(config("bandwidth"), 2.0)))
      }

      state = self.state ++ otherKernel.state

      override def gradientAt(config: Map[String, Double])(x: Index, y: Index): Map[String, Double] = {
        val arg = self.evaluateAt(config)(x,y) +
          self.evaluateAt(config)(y,y) -
          2.0*self.evaluateAt(config)(x,y)

        val gradx = self.gradientAt(config)(x,x)
        val grady = self.gradientAt(config)(y,y)
        val gradxy = self.gradientAt(config)(x,y)

        Map("bandwidth" ->
          otherKernel.evaluateAt(config)(x,y)*arg/math.pow(math.abs(config("bandwidth")), 3)
        ) ++
          gradxy.map((s) => {
            val ans = (-2.0*s._2 + gradx(s._1) + grady(s._1))/2.0*math.pow(config("bandwidth"), 2.0)
            (s._1, -1.0*otherKernel.evaluateAt(config)(x,y)*ans)
          })
      }
      
      override def setHyperParameters(h: Map[String, Double]) = {
        self.setHyperParameters(h)
        otherKernel.setHyperParameters(h)
        super.setHyperParameters(h)
      }

      override def buildKernelMatrix[S <: Seq[Index]](mappedData: S, length: Int) =
        SVMKernel.buildSVMKernelMatrix(mappedData, length, this.evaluate)

      override def buildCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
        SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

    }
  }

  def :*[T1](otherKernel: LocalScalarKernel[T1]): KroneckerProductKernel[Index, T1] =
    new KroneckerProductKernel[Index, T1](this, otherKernel)

  def :+[T1](otherKernel: LocalScalarKernel[T1]): CompositeCovariance[(Index, T1)] =
    new TensorCombinationKernel[Index, T1](this, otherKernel)(Reducer.:+:)

  override def buildKernelMatrix[S <: Seq[Index]](mappedData: S, length: Int): KernelMatrix[DenseMatrix[Double]] =
    SVMKernel.buildSVMKernelMatrix[S, Index](mappedData, length, this.evaluate)

  override def buildCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
    SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

  def buildBlockedKernelMatrix[S <: Seq[Index]](mappedData: S, length: Long): PartitionedPSDMatrix =
    SVMKernel.buildPartitionedKernelMatrix(mappedData, length, rowBlocking, colBlocking, this.evaluate)

  def buildBlockedCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
    SVMKernel.crossPartitonedKernelMatrix(dataset1, dataset2, rowBlocking, colBlocking, this.evaluate)

  /**
    * Get a pipeline which when given a particular
    * configuration of hyper-parameters returns this kernel function
    * set with that configuration.
    * */
  def asPipe: DataPipe[Map[String, Double], LocalScalarKernel[Index]] =
    DataPipe((config: Map[String, Double]) => {
      CovarianceFunction(evaluateAt _)(config)
    })

}

abstract class CompositeCovariance[T]
  extends LocalSVMKernel[T] {
  override def repr: CompositeCovariance[T] = this
}

object CompositeCovariance {

  val truncateHyperParams = (kv: (String, Double)) => (kv._1.split("/").tail.mkString("/"), kv._2)
}

/**
  * @author mandar2812 date: 22/01/2017
  *
  * A kernel represented as a dot product of an explicit feature mapping.
  *
  * @param p Feature map to be applied on input.
  * */
class FeatureMapCovariance[T, U](p: DataPipe[T, U])(implicit e: InnerProductSpace[U, Double])
  extends LocalSVMKernel[T] { self =>

  val phi = p

  override val hyper_parameters = List.empty[String]

  override def evaluateAt(config: Map[String, Double])(x: T, y: T) = e.dot(phi(x), phi(y))

  /**
    * Construct a multi-layer kernel
    * */
  def >(other: LocalScalarKernel[U]): CompositeCovariance[T] =
    new CompositeCovariance[T] {

      override val hyper_parameters = other.hyper_parameters

      blocked_hyper_parameters = other.blocked_hyper_parameters

      state = other.state

      override def evaluateAt(config: Map[String, Double])(x: T, y: T) =
        other.evaluateAt(config)(self.phi(x), self.phi(y))

      override def setHyperParameters(h: Map[String, Double]) = {
        other.setHyperParameters(h)
        super.setHyperParameters(h)
      }

      override def gradientAt(config: Map[String, Double])(x: T, y: T) =
        other.gradientAt(config)(self.phi(x), self.phi(y))
    }

  /**
    * Construct a multi-layer feature map kernel
    * */
  def >[V](other: FeatureMapCovariance[U, V])(
    implicit e1: InnerProductSpace[V, Double]): FeatureMapCovariance[T, V] =
    new FeatureMapCovariance[T, V](self.phi > other.phi)

}


