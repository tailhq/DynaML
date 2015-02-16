/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.kuleuven.esat.kernels

import breeze.linalg.{DenseMatrix, DenseVector}

/**
 * Standard Polynomial SVM Kernel
 * of the form K(Xi,Xj) = (Xi^T * Xj + d)^r
 */
class PolynomialKernel(
    private var degree: Int,
    private var offset: Double)
  extends SVMKernel[DenseMatrix[Double]]
  with Serializable{

  def setDegree(d: Int): Unit = {
    this.degree = d
  }

  def setOffset(o: Int): Unit = {
    this.offset = o
  }

  override def evaluate(x: DenseVector[Double], y: DenseVector[Double]): Double =
    Math.pow(x dot y + this.offset, this.degree)

  override def buildKernelMatrix(
      mappedData: List[DenseVector[Double]],
      length: Int): KernelMatrix[DenseMatrix[Double]] =
    SVMKernel.buildSVMKernelMatrix(mappedData, length, this.evaluate)
}
