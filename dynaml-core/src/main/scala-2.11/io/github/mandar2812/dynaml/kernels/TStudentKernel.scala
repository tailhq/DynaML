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

import breeze.linalg.{norm, DenseVector, DenseMatrix}

/**
  * T-Student Kernel
  * K(x,y) = 1/(1 + ||x - y||<sup>d</sup>)
  */
class TStudentKernel(private var d: Double = 1.0)
  extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {

  override val hyper_parameters = List("d")

  state = Map("d" -> d)

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val diff = x - y
    1.0/(1.0 + math.pow(norm(diff, 2), state("d")))
  }

  def getD: Double = state("d")

}

class TStudentCovFunc(private var d: Double) extends LocalSVMKernel[Double] {
  override val hyper_parameters: List[String] = List("d")

  state = Map("d" -> d)

  override def evaluate(x: Double, y: Double): Double =
    1.0/(1.0 + math.pow(math.abs(x-y), state("d")))
}

class CoRegTStudentKernel(bandwidth: Double) extends LocalSVMKernel[Int] {

  override val hyper_parameters: List[String] = List("d")

  state = Map("d" -> bandwidth)

  override def evaluate(x: Int, y: Int): Double = {
    val diff = x - y
    1.0/(1.0 + math.pow(math.abs(diff.toDouble), state("d")))
  }

  def getD: Double = state("d")
}
