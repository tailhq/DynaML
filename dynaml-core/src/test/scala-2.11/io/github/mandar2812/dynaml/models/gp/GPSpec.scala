package io.github.mandar2812.dynaml.models.gp

import breeze.linalg._
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.algebra.PartitionedVector
import io.github.mandar2812.dynaml.kernels.{DiracKernel, PolynomialKernel, SEKernel}
import io.github.mandar2812.dynaml.pipes.DataPipe
import org.scalatest.{FlatSpec, Matchers}

class GPSpec extends FlatSpec with Matchers {

  "A gaussian process regression model" should " compute likelihood correctly" in {

    val epsilon = 1E-5

    val data = Seq((DenseVector(0d), 0d), (DenseVector(1d), 1d))

    implicit val field = VectorField(1)
    implicit val tr    = identityPipe[Seq[(DenseVector[Double], Double)]]

    val seKernel = new DiracKernel(noiseLevel = 0.5d)

    val mean = DataPipe[DenseVector[Double], Double](v => v(0))

    val noiseCov = new DiracKernel(noiseLevel = 0.5d)

    val gp_model = AbstractGPRegressionModel(
      seKernel, noiseCov,
      mean)(data, data.length)

    val likelihood = gp_model.energy(gp_model._current_state)

    val actual_likelihood = -math.log(0.15915494309189535)

    assert(likelihood - actual_likelihood < epsilon)

  }

  "A gaussian process regression model" should " compute posterior pdf correctly" in {

    val epsilon = 1E-5

    val data = Seq((DenseVector(1d), 1d), (DenseVector(2d), 4d))

    implicit val field = VectorField(1)
    implicit val tr    = identityPipe[Seq[(DenseVector[Double], Double)]]

    val polyKernel = new PolynomialKernel(2, 0.0)
    polyKernel.block("degree")

    val mean = DataPipe[DenseVector[Double], Double](v => 0d)

    val noiseCov = new DiracKernel(noiseLevel = 0.5d)

    val gp_model = AbstractGPRegressionModel(
      polyKernel, noiseCov,
      mean)(data.take(1), 1)

    val posterior = gp_model.predictiveDistribution(data.takeRight(1).map(_._1))

    val likelihood = -posterior.underlyingDist.logPdf(PartitionedVector(Stream((0L, data.takeRight(1).head._1))))

    val actual_likelihood = 11.358593590728782

    assert(likelihood - actual_likelihood < epsilon)

  }


}
