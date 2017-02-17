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

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.algebra.PartitionedPSDMatrix
import io.github.mandar2812.dynaml.kernels.{CovarianceFunction, LocalScalarKernel}
import io.github.mandar2812.dynaml.models.lm.{GeneralizedLeastSquaresModel, GeneralizedLinearModel}
import io.github.mandar2812.dynaml.pipes.{DataPipe, DataPipe2, DataPipe3}

/**
  * Created by mandar on 15/6/16.
  */
class GLMPipe[T, Source](pre: (Source) => Stream[(DenseVector[Double], Double)],
                         map: (DenseVector[Double]) => (DenseVector[Double]) = identity _,
                         task: String = "regression", modelType: String = "") extends
  ModelPipe[Source, Stream[(DenseVector[Double], Double)],
    DenseVector[Double], Double,
    GeneralizedLinearModel[T]] {

  override val preProcess = pre

  override def run(data: Source) = {
    val training = preProcess(data)
    GeneralizedLinearModel[T](training, task, map, modelType)
  }

}

class GLMPipe2[T, Source](
  pre: (Source) => Stream[(DenseVector[Double], Double)],
  task: String = "regression", modelType: String = "") extends DataPipe2[
  Source, DataPipe[DenseVector[Double], DenseVector[Double]],
  GeneralizedLinearModel[T]] {

  override def run(
    data1: Source,
    data2: DataPipe[DenseVector[Double], DenseVector[Double]]) = {

    val training_data = pre(data1)
    GeneralizedLinearModel[T](training_data, task, data2.run, modelType)
  }
}

object GeneralizedLeastSquaresPipe3 extends DataPipe3[
    Stream[(DenseVector[Double], Double)],
    PartitionedPSDMatrix,
    DataPipe[DenseVector[Double], DenseVector[Double]],
    GeneralizedLeastSquaresModel] {

  override def run(
    data1: Stream[(DenseVector[Double], Double)],
    data2: PartitionedPSDMatrix,
    data3: DataPipe[DenseVector[Double], DenseVector[Double]]) =
    new GeneralizedLeastSquaresModel(data1, data2, data3.run)
}

class GeneralizedLeastSquaresPipe2(
  covarianceFunction: LocalScalarKernel[DenseVector[Double]]) extends DataPipe2[
    Stream[(DenseVector[Double], Double)],
    DataPipe[DenseVector[Double], DenseVector[Double]],
    GeneralizedLeastSquaresModel] {


  override def run(
    data1: Stream[(DenseVector[Double], Double)],
    data2: DataPipe[DenseVector[Double], DenseVector[Double]]) = {

    //Construct the covariance matrix from the data
    val Omega = covarianceFunction.buildBlockedKernelMatrix(data1.map(_._1), data1.length.toLong)
    GeneralizedLeastSquaresPipe3(data1, Omega, data2)
  }
}
