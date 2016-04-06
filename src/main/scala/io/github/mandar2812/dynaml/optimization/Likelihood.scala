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
  * Created by mandar on 6/4/16.
  *
  * Represents a likelihood of the form P(q|r)
  *
  * @tparam Q The type of observed variable (q)
  * @tparam R The type of conditioning variable (r)
  */
trait Likelihood[Q, R, S] {

  /**
    * Computes - log p(q|r)
    * */
  def loglikelihood(q: Q, r: R): Double

  /**
    * Computes d(log p(q|r))/dr
    * */
  def gradient(q: Q, r: R): R


  /**
    * Computes d<sup>2</sup>(log p(q|r))/dr<sup>2</sup>
    * */
  def hessian(q: Q, r: R): S

}

/**
  * Represents an IID sigmoid likelihood used
  * in Gaussian Process binary classification models
  *
  * p(q = 1| r) = 1/(1 + exp(-qr))
  *
  * */
class VectorIIDSigmoid extends
  Likelihood[DenseVector[Double],
    DenseVector[Double],
    DenseMatrix[Double]] {

  /**
    * Computes - log p(q|r)
    **/
  override def loglikelihood(q: DenseVector[Double], r: DenseVector[Double]): Double =
    (q.toArray zip r.toArray).map((couple) => {
      math.log(sigmoid(couple._1*couple._2))
    }).sum

  /**
    * Computes d(log p(q|r))/dr
    **/
  override def gradient(q: DenseVector[Double],
                        r: DenseVector[Double]) =
    DenseVector((q.toArray zip r.toArray).map((couple) => {
      ((couple._1 + 1.0)/2.0) - sigmoid(couple._1*couple._2)
    }))

  /**
    * Computes d<sup>2</sup>(log p(q|r))/dr<sup>2</sup>
    **/
  override def hessian(q: DenseVector[Double],
                       r: DenseVector[Double]): DenseMatrix[Double] = {
    diag(DenseVector((q.toArray zip r.toArray).map((couple) => {
      val pi = sigmoid(couple._1*couple._2)
      -1.0*pi*(1.0 - pi)
    })))
  }
}



/**
  * Represents an IID probit likelihood used
  * in Gaussian Process binary classification models
  *
  * p(q = 1| r) = &Phi;(qr)
  *
  * */
class VectorIIDProbit extends
  Likelihood[DenseVector[Double],
    DenseVector[Double],
    DenseMatrix[Double]] {

  private val standardGaussian = new NormalDistribution(0, 1.0)

  val h = (x: Double) =>
    standardGaussian.cumulativeProbability(x)

  /**
    * Computes - log p(q|r)
    **/
  override def loglikelihood(q: DenseVector[Double], r: DenseVector[Double]): Double =
    (q.toArray zip r.toArray).map((couple) => {
      -1.0*math.log(h(couple._1*couple._2))
    }).sum

  /**
    * Computes d(log p(q|r))/dr
    **/
  override def gradient(q: DenseVector[Double],
                        r: DenseVector[Double]) =
    DenseVector((q.toArray zip r.toArray).map((couple) => {
      couple._1*standardGaussian.density(couple._2)/h(couple._1*couple._2)
    }))

  /**
    * Computes d<sup>2</sup>(log p(q|r))/dr<sup>2</sup>
    **/
  override def hessian(q: DenseVector[Double],
                       r: DenseVector[Double]): DenseMatrix[Double] = {
    diag(DenseVector((q.toArray zip r.toArray).map((couple) => {
      val n = standardGaussian.density(couple._2)
      val product = couple._1*couple._2
      val l = h(product)
      -1.0*(n*n)/(l*l) - product*n/l
    })))
  }
}