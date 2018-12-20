package io.github.mandar2812.dynaml.kernels

import breeze.linalg.DenseMatrix

/**
  * A kernel formed by multiplication
  * of two kernels k(.,.) = k1(.,.) * k2(.,.)
  * @author mandar date: 22/01/2017.
  *
  * */
class MultiplicativeCovariance[Index](
  val firstKernel: LocalScalarKernel[Index],
  val otherKernel: LocalScalarKernel[Index])
  extends CompositeCovariance[Index] {

  val (fID, sID) = (firstKernel.toString.split("\\.").last, otherKernel.toString.split("\\.").last)

  override val hyper_parameters =
    firstKernel.hyper_parameters.map(h => fID+"/"+h) ++
      otherKernel.hyper_parameters.map(h => sID+"/"+h)

  protected def getKernelConfigs(config: Map[String, Double]) = (
    config.filter(_._1.contains(fID)).map(CompositeCovariance.truncateState),
    config.filter(_._1.contains(sID)).map(CompositeCovariance.truncateState)
  )

  protected def getKernelHyp(s: Seq[String]) = (
    s.filter(_.contains(fID)).map(CompositeCovariance.truncateHyp),
    s.filter(_.contains(sID)).map(CompositeCovariance.truncateHyp)
  )

  override def evaluateAt(config: Map[String, Double])(x: Index, y: Index) = {

    val (firstKernelConfig, secondKernelConfig) = getKernelConfigs(config)

    firstKernel.evaluateAt(firstKernelConfig)(x,y) * otherKernel.evaluateAt(secondKernelConfig)(x,y)

  }


  state = firstKernel.state.map(h => (fID+"/"+h._1, h._2)) ++ otherKernel.state.map(h => (sID+"/"+h._1, h._2))

  blocked_hyper_parameters =
    firstKernel.blocked_hyper_parameters.map(h => fID+"/"+h) ++
      otherKernel.blocked_hyper_parameters.map(h => sID+"/"+h)

  override def setHyperParameters(h: Map[String, Double]): this.type = {

    val (firstKernelConfig, secondKernelConfig) = getKernelConfigs(h)

    firstKernel.setHyperParameters(firstKernelConfig)
    otherKernel.setHyperParameters(secondKernelConfig)
    super.setHyperParameters(h)
  }

  override def block(h: String*) = {

    val (firstKernelHyp, secondKernelHyp) = getKernelHyp(h)
    firstKernel.block(firstKernelHyp:_*)
    otherKernel.block(secondKernelHyp:_*)
    super.block(h:_*)
  }

  override def gradientAt(config: Map[String, Double])(x: Index, y: Index): Map[String, Double] = {

    val (firstKernelConfig, secondKernelConfig) = getKernelConfigs(config)

    firstKernel.gradientAt(firstKernelConfig)(x, y).map((couple) =>
      (fID+"/"+couple._1, couple._2*otherKernel.evaluateAt(secondKernelConfig)(x,y))
    ) ++ otherKernel.gradientAt(secondKernelConfig)(x,y).map((couple) =>
      (sID+"/"+couple._1, couple._2*firstKernel.evaluateAt(firstKernelConfig)(x,y))
    )
  }

  override def buildKernelMatrix[S <: Seq[Index]](mappedData: S, length: Int) =
    SVMKernel.buildSVMKernelMatrix[S, Index](mappedData, length, this.evaluate)

  override def buildCrossKernelMatrix[S <: Seq[Index]](dataset1: S, dataset2: S) =
    SVMKernel.crossKernelMatrix(dataset1, dataset2, this.evaluate)

}

/**
  * Implementation of separable stationary kernels as defined in
  * http://jmlr.csail.mit.edu/papers/volume2/genton01a/genton01a.pdf
  *
  * K(x,y) = K1(x)&times;K2(y)
  *
  * @tparam I The index set or input domain over which the kernel function is evaluated.
  * @tparam Kernel A kernel type which extends [[StationaryKernel]] as well as [[LocalScalarKernel]]
  * @param firstKernel The kernel given by K1,
  *                    it is assumed that the user inputs a valid stationary kernel of type [[Kernel]]
  * @param otherKernel The kernel given by K2,
  *                    it is assumed that the user inputs a valid stationary kernel of type [[Kernel]]
  * @author mandar2812 date 21/06/2017
  * */
class SeparableStationaryKernel[
I, Kernel <: StationaryKernel[I, Double, DenseMatrix[Double]] with LocalScalarKernel[I]](
  override val firstKernel: Kernel, override val otherKernel: Kernel) extends
  MultiplicativeCovariance[I](firstKernel, otherKernel) {


  override def evaluateAt(config: Map[String, Double])(x: I, y: I) = {

    val (firstKernelConfig, secondKernelConfig) = getKernelConfigs(config)

    firstKernel.evalAt(firstKernelConfig)(x) * otherKernel.evalAt(secondKernelConfig)(y)

  }

  override def gradientAt(config: Map[String, Double])(x: I, y: I) = {

    val (firstKernelConfig, secondKernelConfig) = getKernelConfigs(config)

    firstKernel.gradientAt(firstKernelConfig)(x, x).map((couple) =>
      (fID+"/"+couple._1, couple._2*otherKernel.evaluateAt(secondKernelConfig)(y,y))
    ) ++ otherKernel.gradientAt(secondKernelConfig)(y,y).map((couple) =>
      (sID+"/"+couple._1, couple._2*firstKernel.evaluateAt(firstKernelConfig)(x,x))
    )

  }
}

