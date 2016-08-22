package io.github.mandar2812.dynaml.kernels

/**
  * Created by mandar on 22/08/16.
  */
class MixedEffectRegularizer(om: Double) extends LocalSVMKernel[Int] {

  state = Map("omega" -> om)

  override val hyper_parameters: List[String] = List("omega")

  override def gradient(x: Int, y: Int): Map[String, Double] = Map("omega" -> {if(x == y) 0.0 else 1.0})

  override def evaluate(x: Int, y: Int): Double = if(x == y) 1.0 else state("omega")
}
