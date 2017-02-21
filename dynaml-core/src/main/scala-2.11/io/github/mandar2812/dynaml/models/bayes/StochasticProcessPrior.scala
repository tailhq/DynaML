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
package io.github.mandar2812.dynaml.models.bayes

import io.github.mandar2812.dynaml.models.StochasticProcessModel
import io.github.mandar2812.dynaml.probability.RandomVarWithDistr
import org.apache.spark.annotation.Experimental



/**
  * @author mandar2812 date: 16/02/2017.
  *
  * Represents a stochastic process with a
  * defined prior distribution.
  */
@Experimental
trait StochasticProcessPrior[I, Y, Y1,
D <: RandomVarWithDistr[Y1, _], W,
StochasticModel <: StochasticProcessModel[Seq[(I, Y)], I, Y, W]] {

  def posteriorModel(data: Seq[(I, Y)]): StochasticModel

  def posteriorRV[U <: Seq[I]](data: Seq[(I, Y)])(test: U): W =
    posteriorModel(data).predictiveDistribution(test)

  def priorDistribution[U <: Seq[I]](d: U): D

}

