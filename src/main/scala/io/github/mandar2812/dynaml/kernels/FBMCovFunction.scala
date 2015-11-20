package io.github.mandar2812.dynaml.kernels

/**
  * Created by mandar on 20/11/15.
  */
class FBMCovFunction(private var hurst: Double)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("hurst")

  override def evaluate(x: Double, y: Double): Double = {
    0.5*(math.pow(math.abs(x), 2*hurst) +
      math.pow(math.abs(y), 2*hurst) -
      math.pow(math.abs(x-y), 2*hurst))
  }
}