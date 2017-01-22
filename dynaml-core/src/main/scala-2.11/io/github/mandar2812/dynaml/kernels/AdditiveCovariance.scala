package io.github.mandar2812.dynaml.kernels

/**
  * @author mandar date: 22/01/2017.
  *
  * A kernel formed by addition
  * of two kernels k(.,.) = k1(.,.) + k2(.,.)
  */
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
