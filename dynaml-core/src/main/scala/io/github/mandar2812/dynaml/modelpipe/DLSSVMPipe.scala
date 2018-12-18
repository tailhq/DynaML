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
package io.github.mandar2812.dynaml.modelpipe

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.models.svm.DLSSVM

/**
  * Created by mandar on 15/6/16.
  */
class DLSSVMPipe[Source](pre: (Source) => Stream[(DenseVector[Double], Double)],
                         cov: LocalScalarKernel[DenseVector[Double]],
                         task: String = "regression") extends
  ModelPipe[Source, Stream[(DenseVector[Double], Double)],
    DenseVector[Double], Double, DLSSVM] {

  override val preProcess = pre

  override def run(data: Source) = {
    val training = preProcess(data)
    new DLSSVM(training, training.length, cov, task)
  }
}
