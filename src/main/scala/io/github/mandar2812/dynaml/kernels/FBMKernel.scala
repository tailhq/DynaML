package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{norm, DenseVector, DenseMatrix}

/**
  * Created by mandar on 20/11/15.
  */
class FBMCovFunction(private var hurst: Double)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("hurst")

  state = Map("hurst" -> hurst)

  override def evaluate(x: Double, y: Double): Double = {
    0.5*(math.pow(math.abs(x), 2*state("hurst")) +
      math.pow(math.abs(y), 2*state("hurst")) -
      math.pow(math.abs(x-y), 2*state("hurst")))
  }

  override def gradient(x: Double, y: Double): Map[String, Double] = {
    val a = math.pow(x, 2)
    val b = math.pow(y, 2)
    val c = math.pow(x-y, 2)
    val grad = if(math.log(c) != Double.NaN
      && math.log(a) != Double.NaN
      && math.log(b) != Double.NaN) {
      math.pow(a, state("hurst"))*math.log(a) +
        math.pow(b, state("hurst"))*math.log(b) -
        math.pow(c, state("hurst"))*math.log(c)
    } else if(math.log(c) == Double.NaN){
      Double.NegativeInfinity
    } else {
      Double.PositiveInfinity
    }
    Map("hurst" -> grad)
  }
}

/**
  * Fractional Brownian Kernel:
  *
  * Fractional Brownian Motion is a stochastic
  * process first studied by Mandelbrot and Von Ness
  * its covariance function generalized to multivariate
  * index sets is.
  *
  * K(x,y) = 1/2*(||x||<sup>2H</sup> + ||y||<sup>2H</sup> - ||x-y||<sup>2H</sup>)
  *
  * */
class FBMKernel(private var hurst: Double = 0.75)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {

  override val hyper_parameters = List("hurst")

  state = Map("hurst" -> hurst)

  def sethurst(d: Double): Unit = {
    this.hurst = d
    state += ("hurst" -> d)
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val diff = x - y
    0.5*(math.pow(norm(x, 2), 2*state("hurst")) + math.pow(norm(y, 2), 2*state("hurst")) -
      math.pow(norm(diff, 2), 2*state("hurst")))
  }

  def getHurst: Double = this.hurst

  override def gradient(x: DenseVector[Double],
                        y: DenseVector[Double]): Map[String, Double] = {
    val a = math.pow(norm(x, 2), 2)
    val b = math.pow(norm(y, 2), 2)
    val c = math.pow(norm(x-y, 2), 2)
    val grad = if(math.log(c) != Double.NaN
      && math.log(a) != Double.NaN
      && math.log(b) != Double.NaN) {
      math.pow(a, state("hurst"))*math.log(a) +
        math.pow(b, state("hurst"))*math.log(b) -
        math.pow(c, state("hurst"))*math.log(c)
    } else if(math.log(c) == Double.NaN){
      Double.NegativeInfinity
    } else {
      Double.PositiveInfinity
    }
    Map("hurst" -> grad)
  }
}