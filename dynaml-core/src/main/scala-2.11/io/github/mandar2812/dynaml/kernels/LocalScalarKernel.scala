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

class AdditiveCovariance[Index](
  firstKernel: LocalScalarKernel[Index],
  otherKernel: LocalScalarKernel[Index]) extends CompositeCovariance[Index] {

  val (fID, sID) = (firstKernel.toString.split("\\.").last, otherKernel.toString.split("\\.").last)

  override val hyper_parameters =
    firstKernel.hyper_parameters.map(h => fID+"/"+h) ++
      otherKernel.hyper_parameters.map(h => sID+"/"+h)

  override def evaluate(x: Index, y: Index) = firstKernel.evaluate(x,y) + otherKernel.evaluate(x,y)

  state = firstKernel.state.map(h => (fID+"/"+h._1, h._2)) ++ otherKernel.state.map(h => (sID+"/"+h._1, h._2))

  blocked_hyper_parameters =
    firstKernel.blocked_hyper_parameters.map(h => fID+"/"+h) ++
      otherKernel.blocked_hyper_parameters.map(h => sID+"/"+h)

  override def setHyperParameters(h: Map[String, Double]): this.type = {
    firstKernel.setHyperParameters(h.filter(_._1.contains(fID))
      .map(kv => (kv._1.split("/").tail.mkString("/"), kv._2)))
    otherKernel.setHyperParameters(h.filter(_._1.contains(sID))
      .map(kv => (kv._1.split("/").tail.mkString("/"), kv._2)))
    this
  }

  override def gradient(x: Index, y: Index): Map[String, Double] =
    firstKernel.gradient(x, y) ++ otherKernel.gradient(x,y)

  override def buildKernelMatrix[S <: Seq[Index]](mappedData: S, length: Int) =
    SVMKernel.buildSVMKernelMatrix[S, Index](mappedData, length, this.evaluate)

  override def buildCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
    SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

}

class MultiplicativeCovariance[Index](
  firstKernel: LocalScalarKernel[Index],
  otherKernel: LocalScalarKernel[Index])
  extends CompositeCovariance[Index] {

  val (fID, sID) = (firstKernel.toString.split("\\.").last, otherKernel.toString.split("\\.").last)

  override val hyper_parameters =
    firstKernel.hyper_parameters.map(h => fID+"/"+h) ++
      otherKernel.hyper_parameters.map(h => sID+"/"+h)

  override def evaluate(x: Index, y: Index) = firstKernel.evaluate(x,y) * otherKernel.evaluate(x,y)

  state = firstKernel.state.map(h => (fID+"/"+h._1, h._2)) ++ otherKernel.state.map(h => (sID+"/"+h._1, h._2))

  blocked_hyper_parameters =
    firstKernel.blocked_hyper_parameters.map(h => fID+"/"+h) ++
      otherKernel.blocked_hyper_parameters.map(h => sID+"/"+h)

  override def setHyperParameters(h: Map[String, Double]): this.type = {
    firstKernel.setHyperParameters(h.filter(_._1.contains(fID))
      .map(kv => (kv._1.split("/").tail.mkString("/"), kv._2)))
    otherKernel.setHyperParameters(h.filter(_._1.contains(sID))
      .map(kv => (kv._1.split("/").tail.mkString("/"), kv._2)))
    this
  }

  override def gradient(x: Index, y: Index): Map[String, Double] =
    firstKernel.gradient(x, y).map((couple) => (couple._1, couple._2*otherKernel.evaluate(x,y))) ++
      otherKernel.gradient(x,y).map((couple) => (couple._1, couple._2*firstKernel.evaluate(x,y)))

  override def buildKernelMatrix[S <: Seq[Index]](mappedData: S, length: Int) =
    SVMKernel.buildSVMKernelMatrix[S, Index](mappedData, length, this.evaluate)

  override def buildCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
    SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

}
/**
  * A kernel/covariance function which can be seen as a combination
  * of base kernels over a subset of the input space.
  *
  * for example K((x1, y1), (x1, y2)) = k1(x1,x2) + k2(y1,y2)
  */
class DecomposableCovariance[S](kernels: LocalScalarKernel[S]*)(
  implicit encoding: Encoder[S, Array[S]],
  reducer: Reducer = Reducer.:+:) extends CompositeCovariance[S] {

  val kernelMap = kernels.map(k => (k.toString.split("\\.").last, k)).toMap

  state = kernels.map(k => {
    val id = k.toString.split("\\.").last
    k.state.map(h => (id+"/"+h._1, h._2))
  }).reduceLeft(_++_)

  val encodingTuple = encoding*encoding

  override val hyper_parameters: List[String] = kernels.map(k => {
    val id = k.toString.split("\\.").last
    k.hyper_parameters.map(h => id+"/"+h)
  }).reduceLeft(_++_)

  blocked_hyper_parameters = kernels.map(k => {
    val id = k.toString.split("\\.").last
    k.blocked_hyper_parameters.map(h => id+"/"+h)
  }).reduceLeft(_++_)

  def kernelBind = DataPipe((xy: (Array[S], Array[S])) => {
    optimize {
      (xy._1, xy._2, kernels.map(k => k.evaluate _ ))
        .zipped
        .map((x, y, k) => k(x, y))
    }
  })

  var kernelPipe = encodingTuple > kernelBind > reducer

  protected def updateKernelPipe(): Unit = kernelPipe = encodingTuple > kernelBind > reducer

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
    updateKernelPipe()
    this
  }

  override def evaluate(x: S, y: S): Double = kernelPipe run (x,y)

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
  }
}

/**
  * Represents a kernel on a product space [[R]] &times [[S]]
  *
  * @param firstK The first covariance
  * @param secondK The second covariance
  * @param reducer An implicit parameter indicating how to combine the
  *                kernel values; it can only be [[Reducer.:+:]] or [[Reducer.:*:]]
  * */
class TensorCombinationKernel[R, S](
  firstK: LocalScalarKernel[R],
  secondK: LocalScalarKernel[S])(implicit reducer: Reducer = Reducer.:*:)
  extends CompositeCovariance[(R,S)] {

  val fID = firstK.toString.split("\\.").last
  val sID = secondK.toString.split("\\.").last

  override val hyper_parameters: List[String] =
    firstK.hyper_parameters.map(h => fID+"/"+h) ++ secondK.hyper_parameters.map(h => sID+"/"+h)

  blocked_hyper_parameters =
    firstK.blocked_hyper_parameters.map(h => fID+"/"+h) ++ secondK.blocked_hyper_parameters.map(h => sID+"/"+h)

  state =
    firstK.state.map(h => (fID+"/"+h._1, h._2)) ++ secondK.state.map(h => (sID+"/"+h._1, h._2))

  override def evaluate(x: (R, S), y: (R, S)): Double =
    reducer(Array(firstK.evaluate(x._1, y._1), secondK.evaluate(x._2, y._2)))

  override def repr: TensorCombinationKernel[R, S] = this

  override def setHyperParameters(h: Map[String, Double]): TensorCombinationKernel.this.type = {
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
      if(kid == fID) firstK.setHyperParameters(hyper_params) else secondK.setHyperParameters(hyper_params)
    })
    this
  }

  override def gradient(x: (R, S), y: (R, S)): Map[String, Double] = reducer match {
    case SumReducer =>
      firstK.gradient(x._1, y._1) ++ secondK.gradient(x._2, y._2)
    case ProductReducer =>
      firstK.gradient(x._1, y._1).map(k => (k._1, k._2*secondK.evaluate(x._2, y._2))) ++
        secondK.gradient(x._2, y._2).map(k => (k._1, k._2*firstK.evaluate(x._1, y._1)))
  }
}
