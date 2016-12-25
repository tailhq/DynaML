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

/**
  * @author mandar2812 on 6/4/16.
  *
  * Represents a likelihood of the form P(y|f)
  *
  * @tparam Q The type of observed variable (y)
  * @tparam R The type of conditioning variable (f)
  * @tparam S Result type of the hessian method.
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
