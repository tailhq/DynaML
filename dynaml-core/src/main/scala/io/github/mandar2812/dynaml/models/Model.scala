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
package io.github.mandar2812.dynaml.models

import io.github.mandar2812.dynaml.optimization._


trait Predictor[Q, R] {

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    * */
  def predict(point: Q): R
}



/**
  * Basic Higher Level abstraction
  * for Machine Learning models.
  *
  * @tparam T The type of the training & test data
  *
  * @tparam Q The type of a single input pattern
  *
  * @tparam R The type of a single output pattern
  *
  * */
trait Model[T, Q, R] extends Predictor[Q, R] {

  /**
    * The training data
    * */
  protected val g: T

  def data: T = g

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    * */
  override def predict(point: Q): R
}

/**
 * Skeleton of Parameterized Model
 * @tparam G The type of the underlying data.
 * @tparam T The type of the parameters
 * @tparam Q A Vector/Matrix representing the features of a point
 * @tparam R The type of the output of the predictive model
 *           i.e. A Real Number or a Vector of outputs.
 * @tparam S The type of the edge containing the
 *           features and label.
 *
 * */
trait ParameterizedLearner[G, T, Q, R, S]
  extends Model[G, Q, R] {
  protected var params: T
  protected val optimizer: RegularizedOptimizer[T, Q, R, S]
  /**
   * Learn the parameters
   * of the model.
   *
   * */
  def learn(): Unit

  /**
   * Get the value of the parameters
   * of the model.
   * */
  def parameters() = this.params

  def updateParameters(param: T): Unit = {
    this.params = param
  }

  def setMaxIterations(i: Int): this.type = {
    this.optimizer.setNumIterations(i)
    this
  }

  def setBatchFraction(f: Double): this.type = {
    assert(f >= 0.0 && f <= 1.0, "Mini-Batch Fraction should be between 0.0 and 1.0")
    this.optimizer.setMiniBatchFraction(f)
    this
  }

  def setLearningRate(alpha: Double): this.type = {
    this.optimizer.setStepSize(alpha)
    this
  }

  def setRegParam(r: Double): this.type = {
    this.optimizer.setRegParam(r)
    this
  }

  def initParams(): T

}

/**
 * Represents skeleton of a
 * Linear Model.
 *
 * @tparam T The underlying type of the data structure
 *           ex. Gremlin, Neo4j, Spark RDD etc
 * @tparam P A Vector/Matrix of Doubles
 * @tparam Q A Vector/Matrix representing the features of a point
 * @tparam R The type of the output of the predictive model
 *           i.e. A Real Number or a Vector of outputs.
 * @tparam S The type of the data containing the
 *           features and label.
 * */

trait LinearModel[T, P, Q , R, S]
  extends ParameterizedLearner[T, P, Q, R, S] {

  /**
    * The non linear feature mapping implicitly
    * defined by the kernel applied, this is initialized
    * to an identity map.
    * */
  var featureMap: Q => Q = identity



  def clearParameters(): Unit

}
