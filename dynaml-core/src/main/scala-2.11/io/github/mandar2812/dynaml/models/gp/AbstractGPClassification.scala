/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
* */
package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.{DenseMatrix, DenseVector, det, inv}
import breeze.numerics._
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.models.{ParameterizedLearner, SecondOrderProcess}
import io.github.mandar2812.dynaml.optimization.{GloballyOptimizable, LaplacePosteriorMode}
import io.github.mandar2812.dynaml.probability.Likelihood

/**
  * @author mandar on 6/4/16.
  *
  * Skeleton of a Gaussian Process binary classification
  * model.
  *
  * @tparam T The type of data structure holding the training instances
  *
  * @tparam I The type of input features (also called index set)
  *
  */
abstract class AbstractGPClassification[T, I](
  data: T, kernel: LocalScalarKernel[I],
  likelihood: Likelihood[DenseVector[Double], DenseVector[Double],
    DenseMatrix[Double], (DenseVector[Double], DenseVector[Double])])
  extends SecondOrderProcess[T, I, Double, Double, DenseMatrix[Double], DenseVector[Double]]
    with ParameterizedLearner[T, DenseVector[Double], I,
    Double, (DenseMatrix[Double], DenseVector[Double])]
    with GloballyOptimizable {


  override protected val g: T = data

  val npoints = dataAsSeq(g).length

  override def initParams() = DenseVector.zeros[Double](npoints)

  override protected var params: DenseVector[Double] = initParams()

  override val mean: (I) => Double = _ => 0

  override val covariance = kernel

  override protected val optimizer = new LaplacePosteriorMode[I](likelihood)

  /**
    * Learn the Laplace approximation
    * to the posterior mean of the nuisance
    * function f(x) ~ GP(m(x), K(x, x'))
    * Here we learn p(f|X,y) by approximating
    * it with a gaussian N(mean(f), Var(f))
    *
    * Note that the routine optimize() only
    * returns the value of mean(f)
    **/
  override def learn(): Unit = {
    val procdata = dataAsIndexSeq(g)
    val targets = DenseVector(dataAsSeq(g).map(_._2).toArray)
    val kernelMat = covariance.buildKernelMatrix(
      procdata, procdata.length)
      .getKernelMatrix()
    params = optimizer.optimize(
      npoints,
      (kernelMat, targets),
      initParams()
    )
  }

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    * Returns p(y = 1| x)
    *
    **/
  override def predict(point: I): Double =
    predictiveDistribution(Seq(point))(0)

  /**
    * Set the model "state" which
    * contains values of its hyper-parameters
    * with respect to the covariance and noise
    * kernels.
    * */
  def setState(s: Map[String, Double]): this.type = {
    covariance.setHyperParameters(s)
    current_state = covariance.state
    this
  }

  override protected var hyper_parameters: List[String] = covariance.hyper_parameters
  override protected var current_state: Map[String, Double] = covariance.state

  /**
    * Calculates the energy of the configuration,
    * in most global optimization algorithms
    * we aim to find an approximate value of
    * the hyper-parameters such that this function
    * is minimized.
    *
    * @param h       The value of the hyper-parameters in the configuration space
    * @param options Optional parameters about configuration
    * @return Configuration Energy E(h)
    **/
  override def energy(h: Map[String, Double],
                      options: Map[String, String]): Double = {

    setState(h)

    val training = dataAsIndexSeq(g)
    val trainingLabels = DenseVector(dataAsSeq(g).map(_._2).toArray)

    val kernelTraining: DenseMatrix[Double] =
      covariance.buildKernelMatrix(training, npoints).getKernelMatrix()

    AbstractGPClassification.logLikelihood(trainingLabels,
      kernelTraining, params,
      optimizer.likelihood)
  }
}

object AbstractGPClassification {

  /**
    * Calculate the marginal log likelihood
    * of the training data for a pre-initialized
    * kernel and noise matrices.
    *
    * @param trainingData The function values assimilated as a [[DenseVector]]
    *
    * @param kernelMatrix The kernel matrix of the training features
    *
    * @param f The estimation of mean(f); from the Laplace approximation
    * */
  def logLikelihood(trainingData: DenseVector[Double],
                    kernelMatrix: DenseMatrix[Double],
                    f : DenseVector[Double],
                    likelihood: Likelihood[
                      DenseVector[Double],
                      DenseVector[Double],
                      DenseMatrix[Double],
                      (DenseVector[Double],
                        DenseVector[Double])]): Double = {

    val kernelTraining: DenseMatrix[Double] = kernelMatrix
    val Kinv = inv(kernelTraining)

    val wMat = likelihood.hessian(trainingData, f) * -1.0
    val wMatsq = sqrt(wMat)

    0.5*(f.t * (Kinv * f) + math.log(det(kernelTraining))) -
      likelihood.loglikelihood(trainingData, f)
  }

}