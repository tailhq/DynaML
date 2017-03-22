package io.github.mandar2812.dynaml.graph

/**
  * A Neural Graph is an encapsulation
  * of a base graph of type [[B]] with
  * a set of predifined behaviors.
  *
  * @tparam B The type of the encapsulated
  *           base graph object
  *
  * @tparam I Type of input
  * @tparam J Type of output
  * */
trait NeuralGraph[B, I, J] {
  protected val g: B
  val forwardPass: (I) => J
}
