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
package io.github.tailhq.dynaml.kernels

import breeze.linalg.{DenseMatrix, DenseVector}
import spire.algebra.Field

/**
  * @author tailhq date: 24/11/15.
  */
class WaveletKernel(func: (Double) => Double)(scale: Double)(implicit ev: Field[DenseVector[Double]])
  extends StationaryKernel[DenseVector[Double], Double, DenseMatrix[Double]]
    with SVMKernel[DenseMatrix[Double]]
    with LocalSVMKernel[DenseVector[Double]]
    with Serializable {

  override val hyper_parameters = List("scale")

  state = Map("scale" -> scale)

  val mother: (Double) => Double = func

  def setscale(d: Double): Unit = {
    state += ("scale" -> d)
  }

  override def evalAt(config: Map[String, Double])(x: DenseVector[Double]): Double =
    x.map(i => mother(math.abs(i)/config("scale"))).toArray.product

  def getscale: Double = state("scale")


}

class WaveletCovFunc(func: (Double) => Double)(scale: Double)
  extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("scale")

  val mother: (Double) => Double = func

  state = Map("scale" -> scale)

  override def evaluateAt(config: Map[String, Double])(x: Double, y: Double): Double =
    mother(math.abs(x-y)/config("scale"))
}
