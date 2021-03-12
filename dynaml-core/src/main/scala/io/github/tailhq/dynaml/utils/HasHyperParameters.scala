package io.github.tailhq.dynaml.utils

/**
  * @author tailhq date 17/05/2017.
  * */
abstract class HasHyperParameters {

  val hyper_parameters: List[String]

  var blocked_hyper_parameters: List[String] = List()

  var state: Map[String, Double] = Map()

  def block(h: String*) = blocked_hyper_parameters = List(h:_*)

  def block_all_hyper_parameters: Unit = {
    blocked_hyper_parameters = hyper_parameters
  }

  def effective_state:Map[String, Double] =
    state.filterNot(h => blocked_hyper_parameters.contains(h._1))

  def effective_hyper_parameters: List[String] =
    hyper_parameters.filterNot(h => blocked_hyper_parameters.contains(h))

  def setHyperParameters(h: Map[String, Double]): this.type = {
    assert(effective_hyper_parameters.forall(h.contains),
      "All hyper parameters must be contained in the arguments")
    effective_hyper_parameters.foreach((key) => {
      state += (key -> h(key))
    })
    this
  }


}
