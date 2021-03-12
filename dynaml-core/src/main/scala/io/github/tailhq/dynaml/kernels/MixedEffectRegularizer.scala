package io.github.tailhq.dynaml.kernels

/**
  * Created by mandar on 22/08/16.
  */
class MixedEffectRegularizer(om: Double) extends LocalScalarKernel[Int] {

  state = Map("omega" -> om)

  override val hyper_parameters: List[String] = List("omega")

  override def gradientAt(config: Map[String, Double])(x: Int, y: Int): Map[String, Double] =
    Map("omega" -> {if(x == y) 0.0 else 1.0})

  override def evaluateAt(config: Map[String, Double])(x: Int, y: Int): Double =
    if(x == y) 1.0 else config("omega")
}
