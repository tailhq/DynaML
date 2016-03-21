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
package io.github.mandar2812.dynaml.pipes

import breeze.linalg.{DenseVector, DenseMatrix}
import io.github.mandar2812.dynaml.kernels.CovarianceFunction
import io.github.mandar2812.dynaml.models._
import io.github.mandar2812.dynaml.models.gp.AbstractGPRegressionModel

/**
  * Top level trait for Pipes involving ML models.
  */
trait ModelPipe[Source, T, Q, R, M <: Model[T, Q, R]]
  extends DataPipe[Source, M]{

  val preProcess: (Source) => T

  override def run(data: Source): M
}


class GPRegressionPipe[M <:
AbstractGPRegressionModel[Seq[(DenseVector[Double], Double)],
  DenseVector[Double]], Source](pre: (Source) => Seq[(DenseVector[Double], Double)],
                                cov: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
                                n: CovarianceFunction[DenseVector[Double], Double, DenseMatrix[Double]],
                                order: Int, ex: Int)
  extends ModelPipe[Source, Seq[(DenseVector[Double], Double)],
    DenseVector[Double], Double, M] {

  override val preProcess: (Source) => Seq[(DenseVector[Double], Double)] = pre

  override def run(data: Source): M =
    AbstractGPRegressionModel[M](preProcess(data), cov, n, order, ex)

}


