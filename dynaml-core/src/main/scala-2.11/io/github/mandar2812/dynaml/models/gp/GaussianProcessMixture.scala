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

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.algebra.{PartitionedPSDMatrix, PartitionedVector}
import io.github.mandar2812.dynaml.analysis.InnerProductPV
import io.github.mandar2812.dynaml.models.{ContMixtureErrorBarsModel, StochasticProcessMixtureModel}
import io.github.mandar2812.dynaml.probability.{ContMixtureRVBars, ContinuousDistrMixture, MultGaussianPRV}
import io.github.mandar2812.dynaml.probability.distributions.BlockedMultiVariateGaussian
import org.apache.log4j.Logger
import spire.algebra.InnerProductSpace

import scala.reflect.ClassTag

/**
  * Represents a multinomial mixture of GP models
  * @tparam I The index set (input domain) over which each component GP is
  *           defined.
  *
  * @author mandar2812 date 14/06/2017.
  * */
class GaussianProcessMixture[T, I: ClassTag](
  override val component_processes: Seq[AbstractGPRegressionModel[T, I]],
  override val weights: DenseVector[Double]) extends
  ContMixtureErrorBarsModel[
    T, I, Double, PartitionedVector, PartitionedPSDMatrix, BlockedMultiVariateGaussian,
    MultGaussianPRV, AbstractGPRegressionModel[T, I]](component_processes, weights) {

  private val logger = Logger.getLogger(this.getClass)

  protected val blockSize: Int = component_processes.head._blockSize

  override protected def toStream(y: PartitionedVector): Stream[Double] = y.toStream

  override protected def getVectorSpace(num_dim: Int): InnerProductSpace[PartitionedVector, Double] =
    InnerProductPV(num_dim, blockSize)


}
