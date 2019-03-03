package io.github.mandar2812.dynaml.models.stp


import breeze.linalg._
import io.github.mandar2812.dynaml.analysis.VectorField
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.kernels.{DiracKernel, SEKernel}
import io.github.mandar2812.dynaml.pipes.DataPipe
import org.scalatest.{FlatSpec, Matchers}

class STPSpec extends FlatSpec with Matchers {

  "A student-t process regression model" should " compute likelihood correctly" in {

    val epsilon = 1E-5

    val data = Seq((DenseVector(0d), 0d), (DenseVector(1d), 1d))

    implicit val field = VectorField(1)
    implicit val tr    = identityPipe[Seq[(DenseVector[Double], Double)]]

    val seKernel = new DiracKernel(noiseLevel = 0.5d)

    val mean = DataPipe[DenseVector[Double], Double](v => v(0))

    val noiseCov = new DiracKernel(noiseLevel = 0.5d)

    val stp_model = AbstractSTPRegressionModel(
      4.0,
      seKernel, noiseCov,
      mean)(data, data.length)

    val likelihood = stp_model.energy(stp_model._current_state)

    val actual_likelihood = -math.log(1/(2.0*math.Pi))

    assert(math.abs(likelihood - actual_likelihood) < epsilon)

  }

  "A student-t process regression model" should " compute posterior pdf and error bars correctly" in {

    val epsilon = 1E-5

    val data = Seq((DenseVector(1d), 1d), (DenseVector(2d), 4d))

    implicit val field = VectorField(1)
    implicit val tr    = identityPipe[Seq[(DenseVector[Double], Double)]]

    val seKernel = new SEKernel(1d, 1d)


    val mean = DataPipe[DenseVector[Double], Double](v => 0d)

    val noiseCov = new DiracKernel(noiseLevel = 0d)
    noiseCov.block_all_hyper_parameters


    val stp_model = AbstractSTPRegressionModel(
      4.0,
      seKernel, noiseCov,
      mean)(data.take(1), 1)

    val posterior = stp_model.predictiveDistribution(data.takeRight(1).map(_._1))

    val (pmean, psigma) = (posterior.underlyingDist.mean, posterior.underlyingDist.variance)

    val preds_with_bars = stp_model.predictionWithErrorBars(data.takeRight(1).map(_._1), 1)


    val rho = math.exp(-1.0/2)

    assert(pmean._data.head._2(0) == rho && math.abs(psigma._data.head._2(0, 0) - (1.0 - rho*rho)*7/5) < epsilon)

    assert(
      preds_with_bars.head._3 == rho - math.sqrt((1.0 - rho*rho)*7/5) &&
        preds_with_bars.head._4 == rho + math.sqrt((1.0 - rho*rho)*7/5))


  }


}
