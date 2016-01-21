package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{diag, DenseMatrix, norm, DenseVector}

/**
 * Standard RBF Kernel of the form
 * K(Xi,Xj) = exp(-||Xi - Xj||**2/2*bandwidth**2)
 */

class RBFKernel(private var bandwidth: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {

  override val hyper_parameters = List("bandwidth")

  state = Map("bandwidth" -> bandwidth)

  def setbandwidth(d: Double): Unit = {
    this.state += ("bandwidth" -> d)
    this.bandwidth = d
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val diff = x - y
    Math.exp(-1*math.pow(norm(diff, 2), 2)/(2*math.pow(this.state("bandwidth"), 2)))
  }

  override def gradient(x: DenseVector[Double],
                        y: DenseVector[Double]): Map[String, Double] =
    Map("bandwidth" -> 1.0*evaluate(x,y)*math.pow(norm(x-y,2),2)/math.pow(math.abs(state("bandwidth")), 3))

  def getBandwidth: Double = this.bandwidth

}

class SEKernel(private var band: Double = 1.0, private var h: Double = 2.0)
  extends RBFKernel(band) {

  state = Map("bandwidth" -> band, "amplitude" -> h)

  override val hyper_parameters = List("bandwidth","amplitude")

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]) =
    math.pow(state("amplitude"), 2.0)*super.evaluate(x,y)

  override def gradient(x: DenseVector[Double],
                        y: DenseVector[Double]): Map[String, Double] =
    Map("amplitude" -> 2.0*state("amplitude")*super.evaluate(x,y)) ++ super.gradient(x,y)

}

class MahalanobisKernel(private var band: DenseVector[Double], private var h: Double = 2.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable  {

  state = Map("MahalanobisAmplitude" -> h) ++
    band.mapPairs((i, b) => ("MahalanobisBandwidth_"+i, b)).toArray.toMap

  override val hyper_parameters = List("MahalanobisAmplitude") ++
    band.mapPairs((i, b) => "MahalanobisBandwidth_"+i).toArray.toList

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]) = {
    val bandMap = state.filter((k) => k._1.contains("MahalanobisBandwidth"))
    assert(x.length == bandMap.size,
      "Mahalanobis Bandwidth vector's must be equal to that of data: "+x.length)
    val diff = x - y
    val bandwidth = DenseMatrix.tabulate[Double](bandMap.size, bandMap.size)((i, j) => {
      if (i == j)
        math.pow(bandMap("MahalanobisBandwidth_"+i), -2.0)
      else
        0.0
    })

    math.pow(state("MahalanobisAmplitude"), 2.0)*
      math.exp((diff.t*(bandwidth*diff))*(-1.0/2.0))
  }

  override def gradient(x: DenseVector[Double],
                        y: DenseVector[Double]): Map[String, Double] = {
    val bandMap = state.filter((k) => k._1.contains("MahalanobisBandwidth"))
    assert(x.length == bandMap.size, "Mahalanobis Bandwidth vector's must be equal to that of data")
    Map("MahalanobisAmplitude" -> 2.0*evaluate(x,y)/state("MahalanobisAmplitude")) ++
      bandMap.map((k) => (k._1, evaluate(x,y)*2.0/math.pow(k._2, 3.0)))
  }

}


class RBFCovFunc(private var bandwidth: Double)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("bandwidth")

  state = Map("bandwidth" -> bandwidth)

  override def evaluate(x: Double, y: Double): Double = {
    val diff = x - y
    Math.exp(-1*math.pow(diff, 2)/(2*math.pow(this.state("bandwidth"), 2)))
  }

  override def gradient(x: Double, y: Double): Map[String, Double] =
    Map("bandwidth" -> evaluate(x,y)*math.pow(x-y,2)/math.pow(math.abs(state("bandwidth")), 3))
}

class SECovFunc(private var band: Double = 1.0, private var h: Double = 2.0)
  extends RBFCovFunc(band) {

  state = Map("bandwidth" -> band, "amplitude" -> h)

  override val hyper_parameters = List("bandwidth","amplitude")

  override def evaluate(x: Double, y: Double) =
    math.pow(state("amplitude"), 2.0)*super.evaluate(x,y)

  override def gradient(x: Double,
                        y: Double): Map[String, Double] =
    Map("amplitude" -> 2.0*state("amplitude")*super.evaluate(x,y)) ++ super.gradient(x,y)

}
