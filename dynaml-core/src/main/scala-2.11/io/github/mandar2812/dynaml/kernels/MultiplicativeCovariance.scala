package io.github.mandar2812.dynaml.kernels

/**
  * @author mandar date: 22/01/2017.
  *
  * A kernel formed by multiplication
  * of two kernels k(.,.) = k1(.,.) * k2(.,.)
  * */
class MultiplicativeCovariance[Index](
  firstKernel: LocalScalarKernel[Index],
  otherKernel: LocalScalarKernel[Index])
  extends CompositeCovariance[Index] {

  val (fID, sID) = (firstKernel.toString.split("\\.").last, otherKernel.toString.split("\\.").last)

  override val hyper_parameters =
    firstKernel.hyper_parameters.map(h => fID+"/"+h) ++
      otherKernel.hyper_parameters.map(h => sID+"/"+h)

  private def getKernelConfigs(config: Map[String, Double]) = (
    config.filter(_._1.contains(fID)).map(CompositeCovariance.truncateHyperParams),
    config.filter(_._1.contains(sID)).map(CompositeCovariance.truncateHyperParams)
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
