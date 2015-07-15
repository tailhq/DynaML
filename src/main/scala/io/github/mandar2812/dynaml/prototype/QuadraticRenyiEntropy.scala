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
package io.github.mandar2812.dynaml.prototype

import breeze.linalg.DenseVector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import io.github.mandar2812.dynaml.kernels.DensityKernel

/**
 * Implements the quadratic Renyi Entropy
 */
class QuadraticRenyiEntropy(dist: DensityKernel)
  extends EntropyMeasure
  with Serializable {

  val log_e = scala.math.log _
  val sqrt = scala.math.sqrt _
  override protected val density: DensityKernel = dist

  /**
   * Calculate the quadratic Renyi entropy
   * within a distribution specific
   * proportionality constant. This can
   * be used to compare the entropy values of
   * different sets of data on the same
   * distribution.
   *
   * @param data The data set whose entropy is
   *             required.
   * @return The entropy of the dataset assuming
   *         it is distributed as given by the value
   *         parameter 'density'.
   * */

  override def entropy(data: List[DenseVector[Double]]): Double = {
    val dim = data.head.length
    val root_two: breeze.linalg.Vector[Double] = DenseVector.fill(dim, sqrt(2))
    val product = for(i <- data.view; j <- data.view) yield (i, j)
    -1*log_e(product.map((couple) =>
      density.eval((couple._1 - couple._2) :/ root_two)).sum)
  }

  override def entropy[K](data: RDD[(K, LabeledPoint)]): Double = {
    val dim = data.first()._2.features.size
    -1*log_e(data.cartesian(data).map((couple) =>{
      val point1: DenseVector[Double] = DenseVector(couple._1._2.features.toArray) / sqrt(2.0)
      val point2: DenseVector[Double] = DenseVector(couple._2._2.features.toArray) / sqrt(2.0)
      density.eval(point1 - point2)
    }).reduce((a,b) => a + b))
  }

  def entropyDifference(entropy: Double,
                        data: List[DenseVector[Double]],
                        add: DenseVector[Double],
                        remove: DenseVector[Double]): Double = {
    val dim = data.head.length
    val expEntropy = math.exp(-1.0*entropy)
    val root_two: breeze.linalg.Vector[Double] = DenseVector.fill(dim, sqrt(2))

    val product1 = for(i <- data.view) yield (remove, i)
    val subtractEnt = product1.map((couple) =>
      density.eval((couple._1 - couple._2) :/ root_two)).sum

    val product2 = for(i <- data.view) yield (add, i)
    val addEnt = product2.map((couple) =>
      density.eval((couple._1 - couple._2) :/ root_two)).sum -
      density.eval((add - remove) :/ root_two)

    -1.0*log_e(expEntropy + addEnt - subtractEnt) - entropy
  }
}
