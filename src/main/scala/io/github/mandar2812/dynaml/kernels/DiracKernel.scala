package io.github.mandar2812.dynaml.kernels

import breeze.linalg.{norm, DenseVector, DenseMatrix}

/**
  * Created by mandar on 1/2/16.
  */
class DiracKernel(private var noiseLevel: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {

  override val hyper_parameters = List("noiseLevel")

  state = Map("noiseLevel" -> noiseLevel)

  def setNoiseLevel(d: Double): Unit = {
    this.state += ("noiseLevel" -> d)
    this.noiseLevel = d
  }

  override def evaluate(x: DenseVector[Double],
                        y: DenseVector[Double]): Double =
    if (norm(x-y, 2) == 0) math.abs(state("noiseLevel"))*1.0 else 0.0

  override def gradient(x: DenseVector[Double],
                        y: DenseVector[Double]): Map[String, Double] =
    Map("noiseLevel" -> 1.0*evaluate(x,y)/math.abs(state("noiseLevel")))

  override def buildKernelMatrix[S <: Seq[DenseVector[Double]]](mappedData: S,
                                                                length: Int)
  : KernelMatrix[DenseMatrix[Double]] =
    new SVMKernelMatrix(DenseMatrix.eye[Double](length)*state("noiseLevel"), length)

}

class MAKernel(private var noiseLevel: Double = 1.0)
  extends LocalSVMKernel[Double]
  with Serializable {
  override val hyper_parameters = List("noiseLevel")

  state = Map("noiseLevel" -> noiseLevel)

  def setNoiseLevel(d: Double): Unit = {
    this.state += ("noiseLevel" -> d)
    this.noiseLevel = d
  }

  override def evaluate(x: Double,
                        y: Double): Double =
    if (x-y == 0.0) math.abs(state("noiseLevel"))*1.0 else 0.0

  override def gradient(x: Double,
                        y: Double): Map[String, Double] =
    Map("noiseLevel" -> 1.0*evaluate(x,y)/math.abs(state("noiseLevel")))

  override def buildKernelMatrix[S <: Seq[Double]](mappedData: S,
                                                   length: Int)
  : KernelMatrix[DenseMatrix[Double]] =
    new SVMKernelMatrix(DenseMatrix.eye[Double](length)*state("noiseLevel"), length)

}