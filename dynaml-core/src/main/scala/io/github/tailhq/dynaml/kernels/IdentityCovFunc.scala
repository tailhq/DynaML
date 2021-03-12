package io.github.tailhq.dynaml.kernels

/**
  * Created by mandar on 23/11/15.
  */
class IdentityCovFunc
  extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List()

  override def evaluate(x: Double, y: Double): Double = if(x == y) 1.0 else 0.0
}