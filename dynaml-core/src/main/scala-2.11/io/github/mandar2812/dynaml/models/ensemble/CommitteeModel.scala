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
package io.github.mandar2812.dynaml.models.ensemble

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.models.gp.GPRegression
import io.github.mandar2812.dynaml.models.neuralnets.FeedForwardNetwork
import io.github.mandar2812.dynaml.models.{LinearModel, Model, ModelPipe}

/**
  * Defines an abstract implementation of a "committee-model".
  *
  * A predictor of the form
  *
  * y(x) = w1*y1(x) + w2*y2(x) + ... + wb*yb(x)
  *
  * is learned, where `(y1(x), y2(x), ..., yb(x))` a set of base models
  * are trained on sub-sampled versions of the training data set and `b`
  * is the number of base models used.
  *
  * @tparam D The type of the data structure containing the
  *           training data set.
  *
  * @tparam D1 The type of data structure containing the data
  *            of the base models.
  *
  * @tparam BaseModel The type of model used as base model
  *                   for the meta model.
  *                   example: [[FeedForwardNetwork]], [[GPRegression]], etc
  *
  * @tparam Pipe A sub-type of [[ModelPipe]] which yields a [[BaseModel]]
  *              with [[D1]] as the base data structure given a
  *              data structure of type [[D]]
  *
  * @param num The number of training data points.
  *
  * @param data The actual training data
  *
  * @param networks A sequence of [[Pipe]] objects yielding [[BaseModel]]
  * */
abstract class CommitteeModel[
D, D1,
BaseModel <: Model[D1, DenseVector[Double], Double],
Pipe <: ModelPipe[D, D1, DenseVector[Double], Double, BaseModel]
](num: Long, data: D, networks: Pipe*) extends
MetaModel[D,D1,BaseModel,Pipe](num, data, networks:_*) with
LinearModel[D, DenseVector[Double], DenseVector[Double], Double, D] {

  val num_points = num


  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: DenseVector[Double]): Double =
    params dot featureMap(point)

  override def clearParameters(): Unit =
    DenseVector.fill[Double](baseNetworks.length)(1.0)

  override def initParams(): DenseVector[Double] =
    DenseVector.fill[Double](baseNetworks.length)(1.0/baseNetworks.length)

  /**
    * Learn the parameters
    * of the model which
    * are in a node of the
    * graph.
    *
    **/
  override def learn(): Unit = {
    params = optimizer.optimize(num_points, g, initParams())
  }

  override protected var params: DenseVector[Double] =
    initParams()

  featureMap = (pattern) =>
    DenseVector(baseNetworks.map(_.predict(pattern)).toArray)


}
