package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.kernels.{DiracKernel, CovarianceFunction}

import scala.annotation.tailrec
import scala.collection.mutable.{MutableList => ML}

/**
  * @author mandar2812 datum 17/11/15.
  *
  * Class representing Gaussian Process regression models
  *
  * y = f(x) + e
  * f(x) ~ GP(0, cov(X,X))
  * e|f(x) ~ N(f(x), noise(X,X))
  */
class GPRegression(
  cov: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
  noise: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]] = new DiracKernel(1.0),
  trainingdata: Seq[(DenseVector[Double], Double)]) extends
AbstractGPRegressionModel[Seq[(DenseVector[Double], Double)],
  DenseVector[Double]](cov, noise, trainingdata,
  trainingdata.length){

  /**
    * Convert from the underlying data structure to
    * Seq[(I, Y)] where I is the index set of the GP
    * and Y is the value/label type.
    **/
  override def dataAsSeq(data: Seq[(DenseVector[Double], Double)]) = data
}

/**
  * @author mandar2812
  *
  * GP-NAR
  * Gaussian Process Non-Linear
  * Auto-regressive Models.
  *
  * y(t) = f(x(t)) + e
  * x(t) = (y(t-1), ... , t-p)
  * f(x) ~ GP(0, cov(X,X))
  * e|f(x) ~ N(0, noise(X,X))
  */
class GPNarModel(order: Int,
                 cov: CovarianceFunction[DenseVector[Double],
                    Double, DenseMatrix[Double]],
                 nL: CovarianceFunction[DenseVector[Double],
                   Double, DenseMatrix[Double]],
                 trainingdata: Seq[(DenseVector[Double], Double)]) extends
GPRegression(cov, nL, trainingdata) {

  val modelOrder = order

  def modelPredictedOutput(n: Int)(input: DenseVector[Double]):
  Seq[(Double, Double, Double)] = {
    assert(modelOrder == input.length, "Model order must be equal to dimension of input")

    @tailrec
    def predictAheadRec(num: Int, features: DenseVector[Double],
                        predictions: ML[(Double, Double, Double)]):
    Seq[(Double, Double, Double)] =
      num match {
        case 0 => predictions.toSeq
        case _ =>
          val pred: (DenseVector[Double], Double, Double, Double) =
            predictionWithErrorBars[Seq[DenseVector[Double]]](Seq(features), 2).head
          val newFeatures = DenseVector(features(1 until modelOrder).toArray ++ Array(pred._2))

          predictAheadRec(num-1, newFeatures, predictions.+=:((pred._2, pred._3, pred._4)))
      }

    predictAheadRec(n, input, ML())
  }
}


/**
  * @author mandar2812
  *
  * GP-NARX
  * Gaussian Process Non-Linear
  * Auto-regressive Model with
  * Exogenous Inputs.
  *
  * y(t) = f(x(t)) + e
  * x(t) = (y(t-1), ... , y(t-p), u(t-1), ..., u(t-p))
  * f(x) ~ GP(0, cov(X,X))
  * e|f(x) ~ N(0, noise(X,X))
  */
class GPNarXModel(order: Int,
                  ex: Int,
                  cov: CovarianceFunction[DenseVector[Double],
                    Double, DenseMatrix[Double]],
                  nL: CovarianceFunction[DenseVector[Double],
                   Double, DenseMatrix[Double]],
                  trainingdata: Seq[(DenseVector[Double], Double)]) extends
GPRegression(cov, nL, trainingdata) {

  val modelOrder = order

  val exogenousInputs = ex

}