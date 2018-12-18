package io.github.mandar2812.dynaml.kernels

import io.github.mandar2812.dynaml.pipes.{ProductReducer, Reducer, SumReducer}

/**
  * Represents a kernel on a product space [[R]] &times; [[S]]
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

  private def getKernelConfigs(config: Map[String, Double]) = (
    config.filter(_._1.contains(fID)).map(CompositeCovariance.truncateState),
    config.filter(_._1.contains(sID)).map(CompositeCovariance.truncateState)
  )

  protected def getKernelHyp(s: Seq[String]) = (
    s.filter(_.contains(fID)).map(CompositeCovariance.truncateHyp),
    s.filter(_.contains(sID)).map(CompositeCovariance.truncateHyp)
  )

  override def evaluateAt(config: Map[String, Double])(x: (R, S), y: (R, S)): Double = {

    val (firstKernelConfig, secondKernelConfig) = getKernelConfigs(config)

    reducer(
      Array(
        firstK.evaluateAt(firstKernelConfig)(x._1, y._1),
        secondK.evaluateAt(secondKernelConfig)(x._2, y._2)
      )
    )
  }

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
    super.setHyperParameters(h)
  }

  override def block(h: String*) = {

    val (firstKernelHyp, secondKernelHyp) = getKernelHyp(h)
    firstK.block(firstKernelHyp:_*)
    secondK.block(secondKernelHyp:_*)
    super.block(h:_*)
  }

  override def gradientAt(config: Map[String, Double])(x: (R, S), y: (R, S)): Map[String, Double] = {
    val (firstKernelConfig, secondKernelConfig) = getKernelConfigs(config)

    reducer match {
      case SumReducer =>
        firstK.gradientAt(firstKernelConfig)(x._1, y._1).map(h => (fID+"/"+h._1, h._2)) ++
          secondK.gradientAt(secondKernelConfig)(x._2, y._2).map(h => (sID+"/"+h._1, h._2))

      case ProductReducer =>
        firstK.gradientAt(firstKernelConfig)(x._1, y._1).map(k =>
          (fID+"/"+k._1, k._2*secondK.evaluateAt(secondKernelConfig)(x._2, y._2))) ++
          secondK.gradientAt(secondKernelConfig)(x._2, y._2).map(k =>
            (sID+"/"+k._1, k._2*firstK.evaluateAt(firstKernelConfig)(x._1, y._1)))

      case _ =>
        super.gradientAt(config)(x, y)
    }
  }
}

class KroneckerProductKernel[R, S](firstK: LocalScalarKernel[R], secondK: LocalScalarKernel[S])
  extends TensorCombinationKernel(firstK, secondK)(Reducer.:*:)