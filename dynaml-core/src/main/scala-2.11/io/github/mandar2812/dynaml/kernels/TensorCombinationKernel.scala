package io.github.mandar2812.dynaml.kernels

import io.github.mandar2812.dynaml.pipes.{ProductReducer, Reducer, SumReducer}

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
    super.setHyperParameters(h)
  }

  override def gradient(x: (R, S), y: (R, S)): Map[String, Double] = reducer match {
    case SumReducer =>
      firstK.gradient(x._1, y._1) ++ secondK.gradient(x._2, y._2)
    case ProductReducer =>
      firstK.gradient(x._1, y._1).map(k => (k._1, k._2*secondK.evaluate(x._2, y._2))) ++
        secondK.gradient(x._2, y._2).map(k => (k._1, k._2*firstK.evaluate(x._1, y._1)))
  }
}
