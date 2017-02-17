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
import io.github.mandar2812.dynaml.modelpipe.ModelPipe
import io.github.mandar2812.dynaml.models.Model
import io.github.mandar2812.dynaml.models.gp.GPRegression
import io.github.mandar2812.dynaml.models.neuralnets.FeedForwardNetwork

/**
  * Defines the basic skeleton of a "meta-model" or
  * a model of models.
  *
  * A set of base models are trained on sub-sampled versions
  * of the training data set and finally a predictor of the form.
  *
  * y(x) = f(y1(x), y2(x), ..., yb(x))
  *
  * Where f is some combination function and
  * b is the number of base models used.
  *
  * @tparam D The type of the data structure containing the
  *           training data set.
  * @tparam D1 The type of data structure containing the data
  *            of the base models.
  * @tparam BaseModel The type of model used as base model
  *                   for the meta model.
  *                   example: [[FeedForwardNetwork]], [[GPRegression]], etc
  * @tparam Pipe A sub-type of [[ModelPipe]] which yields a [[BaseModel]]
  *              with [[D1]] as the base data structure given a
  *              data structure of type [[D]]
  * @param num The number of training data points.
  * @param data The actual training data
  * @param networks A sequence of [[Pipe]] objects yielding [[BaseModel]]
  * */
abstract class MetaModel[
D, D1,
BaseModel <: Model[D1, DenseVector[Double], Double],
Pipe <: ModelPipe[D, D1, DenseVector[Double], Double, BaseModel]
](num: Long, data: D, networks: Pipe*)
  extends Model[D, DenseVector[Double], Double] {

  override protected val g = data

  val baseNetworks: List[BaseModel] =
    networks.toList.map(_(g))

}