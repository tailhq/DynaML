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

import breeze.linalg.{Tensor}
import com.tinkerpop.blueprints.Edge
import com.tinkerpop.frames.EdgeFrame

/**
 * Trait for optimization problem solvers.
 *
 * @tparam P The type of the parameters of the model to be optimized.
 * @tparam Q The type of the predictor variable
 * @tparam R The type of the target variable
 * @tparam S The type of the edge containing the
 *           features and label.
 */
trait Optimizer[P, Q, R, S] extends Serializable {

  /**
   * Solve the convex optimization problem.
   */
  def optimize(nPoints: Long, ParamOutEdges: S, initialP: P): P
}

abstract class RegularizedOptimizer[P, Q, R, S]
  extends Optimizer[P, Q, R, S] with Serializable {

  protected var regParam: Double = 1.0

  protected var numIterations: Int = 10

  protected var miniBatchFraction: Double = 1.0

  protected var stepSize: Double = 1.0

  /**
   * Set the regularization parameter. Default 0.0.
   */
  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  /**
   * Set fraction of data to be used for each SGD iteration.
   * Default 1.0 (corresponding to deterministic/classical gradient descent)
   */
  def setMiniBatchFraction(fraction: Double): this.type = {
    this.miniBatchFraction = fraction
    this
  }

  /**
   * Set the number of iterations for SGD. Default 100.
   */
  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  /**
   * Set the initial step size of SGD for the first step. Default 1.0.
   * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
   */
  def setStepSize(step: Double): this.type = {
    this.stepSize = step
    this
  }
}