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
package io.github.mandar2812.dynaml.probability

import breeze.linalg.{DenseMatrix, DenseVector, diag}
import breeze.numerics.sigmoid

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
