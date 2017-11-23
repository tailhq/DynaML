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

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.utils

/**
  * @author mandar2812 date: 2017/03/08
  *
  * The Neural network kernel.
  *
  *
  * */
class NeuralNetworkKernel(sigma: DenseMatrix[Double]) extends SVMKernel[DenseMatrix[Double]]
  with LocalSVMKernel[DenseVector[Double]]
  with Serializable {

  utils.isSquareMatrix(sigma)

  utils.isSymmetricMatrix(sigma)

  val dimensions = sigma.rows

  state = {
    for(i <- 0 until dimensions; j <- 0 until dimensions)
      yield (i,j)
  }.filter((coup) => coup._1 <= coup._2)
    .map(c => "M_"+c._1+"_"+c._2 -> sigma(c._1, c._2))
    .toMap

  override val hyper_parameters: List[String] = state.keys.toList


  def Sigma(config: Map[String, Double]) = DenseMatrix.tabulate[Double](dimensions, dimensions){(i, j) =>
    if(i <= j) config("M_"+i+"_"+j)
    else config("M_"+j+"_"+i)
  }

  override def evaluateAt(config: Map[String, Double])(x: DenseVector[Double], y: DenseVector[Double]) = {

    val s = Sigma(config)

    val xd = DenseVector(x.toArray ++ Array(1.0))
    val yd = DenseVector(y.toArray ++ Array(1.0))

    val xx: Double = 2.0 * (xd dot (s*xd))
    val yy: Double = 2.0 * (yd dot (s*yd))
    val xy: Double = 2.0 * (xd dot (s*yd))

    2.0*math.sin(xy/math.sqrt((1.0+xx)*(1.0+yy)))/math.Pi

  }
}
