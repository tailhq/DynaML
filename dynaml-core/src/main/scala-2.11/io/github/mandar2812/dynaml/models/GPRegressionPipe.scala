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

import io.github.mandar2812.dynaml.DynaMLPipe
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.models.gp.AbstractGPRegressionModel
import io.github.mandar2812.dynaml.pipes.DataPipe

import scala.reflect.ClassTag

/**
  * Created by mandar on 15/6/16.
  */
class GPRegressionPipe[Source, IndexSet: ClassTag](
  pre: (Source) => Seq[(IndexSet, Double)],
  cov: LocalScalarKernel[IndexSet],
  n: LocalScalarKernel[IndexSet],
  order: Int = 0, ex: Int = 0,
  meanFunc: DataPipe[IndexSet, Double] = DataPipe((_: IndexSet) => 0.0))
  extends ModelPipe[
    Source, Seq[(IndexSet, Double)], IndexSet, Double,
    AbstractGPRegressionModel[Seq[(IndexSet, Double)], IndexSet]] {

  override val preProcess: (Source) => Seq[(IndexSet, Double)] = pre

  implicit val transform = DynaMLPipe.identityPipe[Seq[(IndexSet, Double)]]

  override def run(data: Source): AbstractGPRegressionModel[Seq[(IndexSet, Double)], IndexSet] =
    AbstractGPRegressionModel(cov, n, meanFunc)(preProcess(data), 0)

}


object GPRegressionPipe {
  def apply[Source, IndexSet: ClassTag](
    pre: (Source) => Seq[(IndexSet, Double)],
    cov: LocalScalarKernel[IndexSet], n: LocalScalarKernel[IndexSet],
    order: Int = 0, ex: Int = 0,
    meanFunc: DataPipe[IndexSet, Double] = DataPipe((_: IndexSet) => 0.0)) =
    new GPRegressionPipe[Source, IndexSet](pre, cov, n, order, ex, meanFunc)
}