package io.github.mandar2812.dynaml.kernels

import scalaxy.streams.optimize
import scala.reflect.ClassTag
import breeze.linalg.DenseMatrix
import io.github.mandar2812.dynaml.DynaMLPipe
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
  new DecomposableCovariance(this, otherKernel)(DynaMLPipe.genericReplicationEncoder[Index](2))
  //kernelOps.addLocalScKernels(this, otherKernel)


  /**
    *  Create composite kernel k = k<sub>1</sub> * k<sub>2</sub>
    *
    *  @param otherKernel The kernel to add to the current one.
    *  @return The kernel k defined above.
    *
    * */
  def *[T <: LocalScalarKernel[Index]](otherKernel: T)(implicit ev: ClassTag[Index]): CompositeCovariance[Index] =
  new DecomposableCovariance(this, otherKernel)(
    DynaMLPipe.genericReplicationEncoder[Index](2),
    DecomposableCovariance.:*:)

  def :*[T1](otherKernel: LocalScalarKernel[T1]): CompositeCovariance[(Index, T1)] =
    new KernelOps.PairOps[Index, T1].tensorMultLocalScKernels(this, otherKernel)

  def :+[T1](otherKernel: LocalScalarKernel[T1]): CompositeCovariance[(Index, T1)] =
    new KernelOps.PairOps[Index, T1].tensorAddLocalScKernels(this, otherKernel)

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

/**
  * A kernel/covariance function which can be seen as a combination
  * of base kernels over a subset of the input space.
  *
  * for example K((x1, y1), (x1, y2)) = k1(x1,x2) + k2(y1,y2)
  */
class DecomposableCovariance[S](kernels: LocalScalarKernel[S]*)(
  implicit encoding: Encoder[S, Array[S]],
  reducer: Reducer = DecomposableCovariance.:+:) extends CompositeCovariance[S] {

  val kernelMap = kernels.map(k => (k.toString.split("\\.").last, k)).toMap

  state = kernels.map(k => {
    val id = k.toString.split("\\.").last
    k.state.map(h => (id+"/"+h._1, h._2))
  }).reduceLeft(_++_)

  override val hyper_parameters: List[String] = kernels.map(k => {
    val id = k.toString.split("\\.").last
    k.hyper_parameters.map(h => id+"/"+h)
  }).reduceLeft(_++_)

  blocked_hyper_parameters = kernels.map(k => {
    val id = k.toString.split("\\.").last
    k.blocked_hyper_parameters.map(h => id+"/"+h)
  }).reduceLeft(_++_)

  override def repr: DecomposableCovariance[S] = this

  override def setHyperParameters(h: Map[String, Double]): DecomposableCovariance.this.type = {
    //Sanity Check
    assert(effective_hyper_parameters.forall(h.contains),
      "All hyper parameters must be contained in the arguments")
    //group the hyper params by kernel id
    h.toSeq.filterNot(_._1.split("/").length == 1).map(kv => {
      val idS = kv._1.split("/")
      (idS.head, (idS.tail.mkString("/"), kv._2))
    }).groupBy(_._1).map(hypC => {
      val kid = hypC._1
      val hyper_params = hypC._2.map(_._2).toMap
      kernelMap(kid).setHyperParameters(hyper_params)
    })
    this
  }

  override def evaluate(x: S, y: S): Double = {
    val (xs, ys) = (encoding*encoding)((x,y))
      reducer(
        optimize {
          xs.zip(ys).zip(kernels).map(coupleAndKern => {
            val (u,v) = coupleAndKern._1
            coupleAndKern._2.evaluate(u,v)
          })
        }
      )
  }

  override def gradient(x: S, y: S): Map[String, Double] = reducer match {
    case SumReducer =>
      val (xs, ys) = (encoding*encoding)((x,y))
      xs.zip(ys).zip(kernels).map(coupleAndKern => {
        val (u,v) = coupleAndKern._1
        coupleAndKern._2.gradient(u,v)
      }).reduceLeft(_++_)
    case ProductReducer =>
      val (xs, ys) = (encoding*encoding)((x,y))
      xs.zip(ys).zip(kernels).map(coupleAndKern => {
        val (u,v) = coupleAndKern._1
        coupleAndKern._2.gradient(u,v).mapValues(_ * this.evaluate(x,y)/coupleAndKern._2.evaluate(x,y))
      }).reduceLeft(_++_)
    case _: Reducer =>
      val (xs, ys) = (encoding*encoding)((x,y))
      xs.zip(ys).zip(kernels).map(coupleAndKern => {
        val (u,v) = coupleAndKern._1
        coupleAndKern._2.gradient(u,v)
      }).reduceLeft(_++_)
  }
}

object DecomposableCovariance {

  val :+: = SumReducer

  val :*: = ProductReducer

}