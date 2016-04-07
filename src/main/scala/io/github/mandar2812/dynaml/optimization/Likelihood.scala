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
package io.github.mandar2812.dynaml.optimization

import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.numerics.sigmoid
import org.apache.commons.math3.distribution.NormalDistribution

/**
  * @author mandar2812 on 6/4/16.
  *
  * Represents a likelihood of the form P(y|f)
  *
  * @tparam Q The type of observed variable (y)
  * @tparam R The type of conditioning variable (f)
  * @tparam W The type representing a gaussian distribution for f
  *
  */
trait Likelihood[Q, R, S, W] {

  /**
    * Computes - log p(y|f)
    * */
  def loglikelihood(q: Q, r: R): Double

  /**
    * Computes d(log p(y|f))/dr
    * */
  def gradient(q: Q, r: R): R

  /**
    * Computes d<sup>2</sup>(log p(y|f))/dr<sup>2</sup>
    * */
  def hessian(q: Q, r: R): S

  /**
    * Computes the value of the expectation
    *
    * E<sup>q</sup>[p(y_test = 1| x_test)| X, y]
    *
    * where E<sup>q</sup> denotes (approximate) expectation with
    * respect to the nuisance function distribution.
    *
    * @param normalDistParams The normal approximation to p(f_test| X, y, x_test)
    * */
  def gaussianExpectation(normalDistParams: W): Q

}

/**
  * Represents an IID sigmoid likelihood used
  * in Gaussian Process binary classification models
  *
  * p(y = 1| f) = 1/(1 + exp(-yf))
  *
  * */
class VectorIIDSigmoid extends
  Likelihood[DenseVector[Double],
    DenseVector[Double],
    DenseMatrix[Double],
    (DenseVector[Double], DenseVector[Double])] {

  /**
    * Computes - log p(y|f)
    **/
  override def loglikelihood(y: DenseVector[Double],
                             f: DenseVector[Double]): Double =
    (y.toArray zip f.toArray).map((couple) => {
      math.log(sigmoid(couple._1*couple._2))
    }).sum

  /**
    * Computes d(log p(y|f))/df
    **/
  override def gradient(y: DenseVector[Double],
                        f: DenseVector[Double]) =
    DenseVector((y.toArray zip f.toArray).map((couple) => {
      ((couple._1 + 1.0)/2.0) - sigmoid(couple._1*couple._2)
    }))

  /**
    * Computes d<sup>2</sup>(log p(y|f))/df<sup>2</sup>
    **/
  override def hessian(y: DenseVector[Double],
                       f: DenseVector[Double]): DenseMatrix[Double] = {
    diag(DenseVector((y.toArray zip f.toArray).map((couple) => {
      val pi = sigmoid(couple._1*couple._2)
      -1.0*pi*(1.0 - pi)
    })))
  }

  override def gaussianExpectation(normalDistParams: (DenseVector[Double],
    DenseVector[Double])): DenseVector[Double] = {
    DenseVector((normalDistParams._1.toArray zip normalDistParams._2.toArray).map((couple) => {
      val gamma = math.sqrt(1.0 + (math.Pi*couple._2/8.0))
      sigmoid(couple._1/gamma)
    }))
  }
}



/**
  * Represents an IID probit likelihood used
  * in Gaussian Process binary classification models
  *
  * p(y = 1| f) = &Phi;(yf)
  *
  * */
class VectorIIDProbit extends
  Likelihood[DenseVector[Double],
    DenseVector[Double],
    DenseMatrix[Double],
    (DenseVector[Double], DenseVector[Double])] {

  private val standardGaussian = new NormalDistribution(0, 1.0)

  val h = (x: Double) =>
    standardGaussian.cumulativeProbability(x)

  /**
    * Computes - log p(y|f)
    **/
  override def loglikelihood(y: DenseVector[Double],
                             f: DenseVector[Double]): Double =
    (y.toArray zip f.toArray).map((couple) => {
      -1.0*math.log(h(couple._1*couple._2))
    }).sum

  /**
    * Computes d(log p(y|f))/df
    **/
  override def gradient(y: DenseVector[Double],
                        f: DenseVector[Double]) =
    DenseVector((y.toArray zip f.toArray).map((couple) => {
      couple._1*standardGaussian.density(couple._2)/h(couple._1*couple._2)
    }))

  /**
    * Computes d<sup>2</sup>(log p(y|f))/df<sup>2</sup>
    **/
  override def hessian(y: DenseVector[Double],
                       f: DenseVector[Double]): DenseMatrix[Double] = {
    diag(DenseVector((y.toArray zip f.toArray).map((couple) => {
      val n = standardGaussian.density(couple._2)
      val product = couple._1*couple._2
      val l = h(product)
      -1.0*(n*n)/(l*l) - product*n/l
    })))
  }

  override def gaussianExpectation(normalDistParams: (DenseVector[Double],
    DenseVector[Double])): DenseVector[Double] = {
    DenseVector((normalDistParams._1.toArray zip normalDistParams._2.toArray).map((couple) => {
      val gamma = math.sqrt(1.0 + (math.Pi*couple._2/8.0))
      standardGaussian.cumulativeProbability(couple._1/gamma)
    }))
  }
}