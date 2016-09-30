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
package io.github.mandar2812.dynaml.algebra

import breeze.linalg.{DenseMatrix, DenseVector, eig, max, min}
import org.apache.log4j.{Logger, Priority}

/**
  * @author mandar2812
  *
  */
class SVMKernelMatrix(
    override protected val kernel: DenseMatrix[Double],
    private val dimension: Long)
  extends KernelMatrix[DenseMatrix[Double]]
  with Serializable {
  private val logger = Logger.getLogger(this.getClass)

  /**
   * Calculates the approximate eigen-decomposition of the
   * kernel matrix
   *
   * @param dimensions The effective number of dimensions
   *                   to be calculated in the feature map
   *
   * @return A Scala [[Tuple2]] containing the eigenvalues
   *         and eigenvectors.
   *
   * */
  override def eigenDecomposition(dimensions: Int = this.dimension.toInt):
  (DenseVector[Double], DenseMatrix[Double]) = {
    logger.log(Priority.INFO, "Eigenvalue decomposition of the kernel matrix using JBlas.")
    val decomp = eig(this.kernel)
    logger.log(Priority.INFO, "Eigenvalue stats: "
      +min(decomp.eigenvalues)
      +" =< lambda =< "
      +max(decomp.eigenvalues)
    )
    (decomp.eigenvalues, decomp.eigenvectors)

  }

}
