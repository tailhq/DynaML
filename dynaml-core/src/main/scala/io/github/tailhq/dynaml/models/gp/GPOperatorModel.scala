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
package io.github.tailhq.dynaml.models.gp

import io.github.tailhq.dynaml.algebra.{PartitionedMatrix, PartitionedPSDMatrix}
import io.github.tailhq.dynaml.kernels.{LinearPDEKernel, LocalScalarKernel, SVMKernel}
import io.github.tailhq.dynaml.pipes.DataPipe

import scala.reflect.ClassTag

/**
  * Modified GP formulation for inference of quantities governed
  * by linear PDEs.
  * */
abstract class GPOperatorModel[T, I: ClassTag, K <: LinearPDEKernel[I]](
  cov: K, n: LocalScalarKernel[(I, Double)],
  observations: T, num: Int,
  meanFunc: DataPipe[(I, Double), Double] =
  DataPipe((_:(I, Double)) => 0.0)) extends
  AbstractGPRegressionModel[T, (I, Double)](cov, n, observations, num, meanFunc) {

  override protected def getCrossKernelMatrix[U <: Seq[(I, Double)]](test: U): PartitionedMatrix =
    SVMKernel.crossPartitonedKernelMatrix(
      trainingData, test,
      _blockSize, _blockSize,
      cov.invOperatorKernel)

  override protected def getTestKernelMatrix[U <: Seq[(I, Double)]](test: U): PartitionedPSDMatrix =
    SVMKernel.buildPartitionedKernelMatrix(
      test, test.length.toLong,
      _blockSize, _blockSize,
      cov.baseKernel.evaluate)


}

object GPOperatorModel {

  def apply[T, I: ClassTag, K <: LinearPDEKernel[I]](
    cov: K, noise: LocalScalarKernel[(I, Double)],
    meanFunc: DataPipe[(I, Double), Double])(
    trainingdata: T, num: Int)(
    implicit transform: DataPipe[T, Seq[((I, Double), Double)]]) = {

    val num_points = if(num > 0) num else transform(trainingdata).length

    new GPOperatorModel[T, I, K](cov, noise, trainingdata, num_points, meanFunc) {
      override def dataAsSeq(data: T) = transform(data)
    }

  }
}