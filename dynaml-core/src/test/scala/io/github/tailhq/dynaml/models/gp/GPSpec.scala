package io.github.tailhq.dynaml.models.gp

import breeze.linalg._
import io.github.tailhq.dynaml.analysis.VectorField
import io.github.tailhq.dynaml.DynaMLPipe._
import io.github.tailhq.dynaml.algebra.PartitionedVector
import io.github.tailhq.dynaml.kernels.{DiracKernel, PolynomialKernel, SEKernel}
import io.github.tailhq.dynaml.pipes.DataPipe
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

  "A gaussian process regression model" should " compute posterior pdf and error bars correctly" in {

    val data = Seq((DenseVector(1d), 1d), (DenseVector(2d), 4d))

    implicit val field = VectorField(1)
    implicit val tr    = identityPipe[Seq[(DenseVector[Double], Double)]]

    val seKernel = new SEKernel(1d, 1d)


    val mean = DataPipe[DenseVector[Double], Double](v => 0d)

    val noiseCov = new DiracKernel(noiseLevel = 0d)
    noiseCov.block_all_hyper_parameters


    val gp_model = AbstractGPRegressionModel(
      seKernel, noiseCov,
      mean)(data.take(1), 1)

    val posterior = gp_model.predictiveDistribution(data.takeRight(1).map(_._1))

    val (pmean, psigma) = (posterior.underlyingDist.mean, posterior.underlyingDist.covariance)

    val preds_with_bars = gp_model.predictionWithErrorBars(data.takeRight(1).map(_._1), 1)


    val rho = math.exp(-1.0/2)

    assert(pmean._data.head._2(0) == rho && psigma._data.head._2(0, 0) == 1.0 - rho*rho)

    assert(
      preds_with_bars.head._3 == rho - math.sqrt(1.0 - rho*rho) &&
        preds_with_bars.head._4 == rho + math.sqrt(1.0 - rho*rho))


  }


}
