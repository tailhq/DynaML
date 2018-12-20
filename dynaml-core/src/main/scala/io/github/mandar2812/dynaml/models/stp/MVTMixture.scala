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
package io.github.mandar2812.dynaml.models.stp

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.analysis.MatrixVectorSpace
import io.github.mandar2812.dynaml.models.GenContinuousMixtureModel
import io.github.mandar2812.dynaml.probability.MatrixTRV
import io.github.mandar2812.dynaml.probability.distributions.MatrixT
import spire.algebra.VectorSpace

import scala.reflect.ClassTag

/**
  * @author mandar date 21/06/2017.
  * */
class MVTMixture[T, I: ClassTag](
  override val component_processes: Seq[MVStudentsTModel[T, I]],
  override val weights: DenseVector[Double]) extends
  GenContinuousMixtureModel[
    T, I, DenseVector[Double], DenseMatrix[Double],
    (DenseMatrix[Double], DenseMatrix[Double]), MatrixT,
    MatrixTRV, MVStudentsTModel[T, I]](component_processes, weights) {

  val num_outputs: Int = component_processes.head.num_outputs

  override protected def toStream(y: DenseMatrix[Double]): Stream[DenseVector[Double]] =
    (0 until y.rows).toStream.map(index => y(index,::).t)

  override protected def getVectorSpace(num_dim: Int): VectorSpace[DenseMatrix[Double], Double] =
    MatrixVectorSpace(num_dim, num_outputs)
}
