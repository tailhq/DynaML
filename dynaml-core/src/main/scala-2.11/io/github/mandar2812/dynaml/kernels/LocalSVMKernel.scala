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
package io.github.mandar2812.dynaml.kernels

import breeze.linalg.DenseMatrix
import io.github.mandar2812.dynaml.algebra.PartitionedPSDMatrix

/**
  * Kernels with a locally stored matrix in the form
  * of a breeze [[DenseMatrix]] instance. Optionally
  * a kernel matrix stored as a [[PartitionedPSDMatrix]]
  * can also be generated.
  * */
trait LocalSVMKernel[Index] extends LocalScalarKernel[Index] {
  override def evaluateAt(config: Map[String, Double])(x: Index, y: Index) = {
    println("evaluateAt function not defined, continuing with a stub, "+
      "expect problem with kernel calculation!")
    0.0
  }

  override def gradientAt(config: Map[String, Double])(x: Index, y: Index) = {
    println("gradientAt function not defined, continuing with a stub ")

    effective_hyper_parameters.map((_, 0.0)).toMap
  }

}
