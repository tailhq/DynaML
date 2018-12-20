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
package io.github.mandar2812.dynaml.models.lm

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.mandar2812.dynaml.optimization.{RegularizedLSSolver, RegularizedOptimizer}
import org.apache.spark.rdd.RDD

/**
  * @author mandar2812 date: 25/01/2017.
  * A Generalized Linear Model applied to a
  * single output regression task. The training
  * set is an Apache Spark [[RDD]]
  *
  * @param data The training data as an [[RDD]]
  * @param numPoints Number of training data points
  * @param map A general non-linear feature mapping/basis function expansion.
  *
  */
class SparkGLM(
  data: RDD[(DenseVector[Double], Double)], numPoints: Long,
  map: (DenseVector[Double]) => DenseVector[Double] = identity[DenseVector[Double]])
  extends GenericGLM[
    RDD[(DenseVector[Double], Double)],
    (DenseMatrix[Double], DenseVector[Double])](data, numPoints, map) {

  private lazy val sample_input = g.first()._1

  /**
    * The link function; in this case simply the identity map
    * */
  override val h: (Double) => Double = identity[Double]

  featureMap = map

  override protected var params: DenseVector[Double] = initParams()

  override protected val optimizer: RegularizedOptimizer[
    DenseVector[Double], DenseVector[Double],
    Double, (DenseMatrix[Double], DenseVector[Double])] = new RegularizedLSSolver

  def dimensions = featureMap(sample_input).length

  override def initParams() = DenseVector.zeros[Double](dimensions + 1)

  /**
    * Input an [[RDD]] containing the data set and output
    * a design matrix and response vector which can be solved
    * in the OLS sense.
    * */
  override def prepareData(d: RDD[(DenseVector[Double], Double)]) = {

    val phi = featureMap
    val mapFunc = (xy: (DenseVector[Double], Double)) => {
      val phiX = DenseVector(phi(xy._1).toArray ++ Array(1.0))
      val phiY = phiX*xy._2
      (phiX*phiX.t, phiY)
    }

    d.mapPartitions((partition) => {
      Iterator(partition.map(mapFunc).reduce((a,b) => (a._1+b._1, a._2+b._2)))
    }).reduce((a,b) => (a._1+b._1, a._2+b._2))
  }
}
