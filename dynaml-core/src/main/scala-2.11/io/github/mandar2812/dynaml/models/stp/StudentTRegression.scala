package io.github.mandar2812.dynaml.models.stp

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.evaluation.RegressionMetrics
import io.github.mandar2812.dynaml.kernels.{DiracKernel, LocalScalarKernel}
import io.github.mandar2812.dynaml.pipes.{DataPipe, StreamDataPipe}

/**
  * Created by mandar on 26/08/16.
  */

class StudentTRegression(mu: Double,
  cov: LocalScalarKernel[DenseVector[Double]],
  noise: LocalScalarKernel[DenseVector[Double]] = new DiracKernel(1.0),
  trainingdata: Seq[(DenseVector[Double], Double)]) extends
  AbstractSTPRegressionModel[Seq[(DenseVector[Double], Double)],
    DenseVector[Double]](
    mu, cov, noise, trainingdata,
    trainingdata.length){

  /**
    * Setting a validation set is optional in case
    * one wants to use some function of metrics like correltaion (CC) or rmse (RMSE)
    * as hyper-parameter optimization objective functions.
    * */
  var validationSet: Seq[(DenseVector[Double], Double)] = Seq()

  /**
    * Assigning a value to the [[processTargets]] data pipe
    * can be useful in cases where we need to
    * perform operations such as de-normalizing
    * the predicted and actual targets to their original
    * scales.
    *
    * */
  var processTargets: DataPipe[
    Stream[(Double, Double)],
    Stream[(Double, Double)]] =
  StreamDataPipe((predictionCouple: (Double, Double)) =>
    identity(predictionCouple))

  /**
    * If one uses a non empty validation set, then
    * the user can set a custom function of
    * the validation predictions and targets as
    * the objective function for the hyper-parameter
    * optimization routine.
    *
    * Currently this defaults to RMSE calculated
    * on the validation data.
    * */
  var scoresToEnergy: DataPipe[Stream[(Double, Double)], Double] =
  DataPipe((scoresAndLabels) => {

    val metrics = new RegressionMetrics(
      scoresAndLabels.toList,
      scoresAndLabels.length
    )

    metrics.rmse
  })

  /**
    * Convert from the underlying data structure to
    * Seq[(I, Y)] where I is the index set of the GP
    * and Y is the value/label type.
    **/
  override def dataAsSeq(data: Seq[(DenseVector[Double], Double)]) = data

  /**
    * Calculates the energy of the configuration, required
    * for global optimization routines.
    *
    * Defaults to the base implementation in
    * [[io.github.mandar2812.dynaml.optimization.GloballyOptimizable]]
    * in case a validation set is not specified
    * through the [[validationSet]] variable.
    *
    * @param h The value of the hyper-parameters in the configuration space
    * @param options Optional parameters about configuration
    * @return Configuration Energy E(h)
    * */
  override def energy(h: Map[String, Double],
                      options: Map[String, String]): Double = validationSet.length match {
    case 0 => super.energy(h, options)
    case _ =>
      // Calculate regression metrics on validation set
      // Return some function of kpi as energy
      setState(h)
      val resultsToScores = DataPipe(
        (res: Seq[(DenseVector[Double], Double, Double, Double, Double)]) =>
          res.map(i => (i._3, i._2)).toStream)

      (resultsToScores >
        processTargets >
        scoresToEnergy) run
        this.test(validationSet)
  }

}

