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
package io.github.mandar2812.dynaml.optimization

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Solves the linear problem resulting from applying the Karush-Kuhn-Tucker
  * conditions on the Dual Least Squares SVM optimization problem.
  *
  * @param modelTask Set to "regression" or "classification"
  * @author mandar date 9/2/16.
  *
  * */
class LSSVMLinearSolver(modelTask: String) extends
RegularizedOptimizer[DenseVector[Double],
  DenseVector[Double], Double,
  (DenseMatrix[Double], DenseVector[Double])] {


  var task: String = modelTask

  /**
    * Solve the convex optimization problem.
    *
    * <table border="0">
    *   <tr>
    *     <th>A</th> <th>&nbsp;=&nbsp;</th> <th>K + &gamma;&times;I</th> <th>1</th>
    *   </tr>
    *   <tr>
    *     <th>&nbsp;</th> <th>&nbsp;</th> <th>1<sup>T</sup></th> <th>0</th>
    *   </tr>
    *   <tr height = 20px></tr>
    *   <tr>
    *     <th>b</th> <th>&nbsp;=&nbsp;</th> <th>y</th> <th>&nbsp;</th>
    *   </tr>
    *   <tr>
    *     <th>&nbsp;</th> <th>&nbsp;</th> <th>0</th> <th>&nbsp;</th>
    *   </tr>
    * </table>
    *
    * @param nPoints The number of data points, i.e. also the size of matrix A
    * @param linearSystem The components of the linear system (A, b) as a tuple.
    * @param initialP An initial estimate of the linear system solution, this parameter
    *                 is redundant for [[LSSVMLinearSolver]] as the exact solution is
    *                 computed.
    * @return A<sup>-1</sup>b
    * */
  override def optimize(nPoints: Long,
                        linearSystem: (DenseMatrix[Double], DenseVector[Double]),
                        initialP: DenseVector[Double]): DenseVector[Double] = {

    val (kernelMat,labels) = linearSystem

    val OmegaMat = task match {
      //In case of regression Omega(i,j)  = K(i, j)
      case "regression" => kernelMat
      //In case of classification Omega(i,j)  = y(j)y(j)K(i, j)
      case "classification" => kernelMat *:* (labels * labels.t)
    }


    val smoother = DenseMatrix.eye[Double](initialP.length-1)*regParam
    val ones = DenseMatrix.ones[Double](1,nPoints.toInt)
    //Construct matrix A and b block by block
    val (a,b): (DenseMatrix[Double], DenseVector[Double]) = task match {
      case "regression" =>
        (DenseMatrix.horzcat(
          DenseMatrix.vertcat(OmegaMat + smoother, ones),
          DenseMatrix.vertcat(ones.t, DenseMatrix(0.0))),
          DenseVector.vertcat(labels, DenseVector(0.0)))

      case "classification" =>
        (DenseMatrix.horzcat(
          DenseMatrix.vertcat(OmegaMat + smoother, labels.toDenseMatrix),
          DenseMatrix.vertcat(labels.toDenseMatrix.t, DenseMatrix(0.0))),
          DenseVector.vertcat[Double](ones.toDenseVector, DenseVector(0.0)))
    }

    a\b
  }
}
