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
package io.github.mandar2812.dynaml.models.gp

import breeze.linalg.{DenseMatrix, DenseVector, cholesky, inv}
import breeze.numerics.sqrt
import io.github.mandar2812.dynaml.kernels.LocalScalarKernel
import io.github.mandar2812.dynaml.probability.Likelihood

/**
  * @author mandar2812 on 6/4/16.
  */
class LaplaceBinaryGPC(data: Stream[(DenseVector[Double], Double)],
                       kernel: LocalScalarKernel[DenseVector[Double]],
                       l: Likelihood[
                         DenseVector[Double], DenseVector[Double], DenseMatrix[Double],
                         (DenseVector[Double], DenseVector[Double])]) extends
  AbstractGPClassification[Stream[(DenseVector[Double], Double)],
    DenseVector[Double]](data, kernel, l) {


  /** Calculates posterior predictive distribution for
    * a particular set of test data points.
    *
    * @param test A Sequence or Sequence like data structure
    *             storing the values of the input patters.
    * @return The predictive distribution of the class labels
    *         values at each of the test patterns.
    * */
  override def predictiveDistribution[U <: Seq[DenseVector[Double]]](test: U): DenseVector[Double] = {
    val procdata = dataAsIndexSeq(g)
    val targets = DenseVector(dataAsSeq(g).map(_._2).toArray)
    val kernelMat = covariance.buildKernelMatrix(procdata, procdata.length).getKernelMatrix()

    val wMat = optimizer.likelihood.hessian(targets, params) * -1.0
    val id = DenseMatrix.eye[Double](targets.length)
    val wMatsq = sqrt(wMat)
    val L = cholesky(id + wMatsq*kernelMat*wMatsq)
    val grad = optimizer.likelihood.gradient(targets, params)
    val (meanL, varL) = test.map(point => {
      val features = DenseVector(g.map(inducingpoint =>
        covariance.evaluate(point, inducingpoint._1)).toArray)

      val mean = features dot grad

      val v = inv(L)*(wMatsq*features)
      val variance = covariance.evaluate(point, point) - (v dot v)
      (mean,variance)
    }).unzip

    val nuisanceDist = (DenseVector(meanL.toArray), DenseVector(varL.toArray))

    optimizer.likelihood.gaussianExpectation(nuisanceDist)
  }

  /**
    * Convert from the underlying data structure to
    * Seq[(I, Y)] where I is the index set of the GP
    * and Y is the value/label type.
    **/
  override def dataAsSeq(data: Stream[(DenseVector[Double], Double)]) = data
}
