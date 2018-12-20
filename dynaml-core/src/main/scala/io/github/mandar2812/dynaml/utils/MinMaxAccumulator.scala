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
package io.github.mandar2812.dynaml.utils

import breeze.linalg.DenseVector
import org.apache.spark.AccumulatorParam

object MinMaxAccumulator extends AccumulatorParam[DenseVector[Double]] {
  def zero(initialValue: DenseVector[Double]): DenseVector[Double] = {
    DenseVector(Double.MaxValue, Double.MinValue)
  }

  def addInPlace(v1: DenseVector[Double], v2: DenseVector[Double]): DenseVector[Double] = {
    v1(0) = math.min(v1(0), v2(0))
    v1(1) = math.max(v1(1), v2(1))
    v1
  }
}