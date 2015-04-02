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

package org.kuleuven.esat.graphicalModels

import breeze.linalg.{DenseMatrix, norm, DenseVector}
import com.tinkerpop.blueprints.Direction
import org.apache.log4j.{Logger, Priority}
import org.kuleuven.esat.kernels.{RBFKernel, SVMKernel, GaussianDensityKernel}
import org.kuleuven.esat.prototype.{QuadraticRenyiEntropy, GreedyEntropySelector}
import org.kuleuven.esat.utils

/**
 * Abstract class implementing kernel feature map
 * extraction functions.
 */
abstract class KernelBayesianModel extends
KernelizedModel[DenseVector[Double], DenseVector[Double], Double, Int, Int] {
  protected val logger = Logger.getLogger(this.getClass)

  protected val featuredims: Int

  override def optimumSubset(M: Int): Unit = {
    //Get the original features of the data
    val features = this.filter((_) => true)
    points = (0 to this.npoints - 1).toList
    if(M < this.npoints) {
      logger.log(Priority.INFO, "Calculating sample variance of the data set")

      //Calculate the column means and variances
      val (mean, variance) = utils.getStats(features)

      //Use the adjusted value of the variance
      val adjvarance:DenseVector[Double] = variance :/= (npoints.toDouble - 1)
      val density = new GaussianDensityKernel

      logger.log(Priority.INFO,
        "Using Silvermans rule of thumb to set bandwidth of density kernel")
      logger.log(Priority.INFO,
        "Std Deviation of the data: "+adjvarance.toString())
      logger.log(Priority.INFO,
        "norm: "+norm(adjvarance))
      density.setBandwidth(DenseVector.tabulate[Double](featuredims - 1){
        i => 1.06*math.sqrt(adjvarance(i))/math.pow(npoints, 0.2)
      })
      logger.log(Priority.INFO,
        "Building low rank appriximation to kernel matrix")

      points = GreedyEntropySelector.subsetSelection(this,
        M,
        new QuadraticRenyiEntropy(density),
        0.0001,
        2000)
    }
  }

  override def applyFeatureMap(): Unit = {
    val edges = this.getParamOutEdges().iterator()
    this.g.getVertex("w").setProperty("slope", this.params)
    while (edges.hasNext) {
      //Get the predictor vertex corresponding to the edge
      val vertex = edges.next().getVertex(Direction.IN)
        .getEdges(Direction.IN, "causes").iterator()
        .next().getVertex(Direction.OUT)

      //Get the original features of the point
      val featurex = vertex.getProperty("value").asInstanceOf[DenseVector[Double]]

      //Get mapped features for the point
      val mappedf = featureMap(List(featurex(0 to featurex.length - 2)))(0)
      val newFeatures = DenseVector.vertcat[Double](mappedf, DenseVector(Array(1.0)))

      //Set a new property in the vertex corresponding to the mapped features
      vertex.setProperty("featureMap", newFeatures)
    }
  }

  override def applyKernel(
      kernel: SVMKernel[DenseMatrix[Double]],
      M: Int = math.sqrt(npoints).toInt): Unit = {

    if(M != this.points.length) {
      this.optimumSubset(M)
    }

    val features_of_points = this.filter(p => this.points.contains(p))
    val kernelMatrix =
      kernel.buildKernelMatrix(features_of_points, M)
    val decomposition = kernelMatrix.eigenDecomposition(M)
    this.featureMap = kernel.featureMapping(decomposition)(features_of_points)
    this.params = DenseVector.ones[Double](decomposition._1.length + 1)
    this.applyFeatureMap()
  }

  def applyRBFKernel(
      kernel: RBFKernel,
      M: Int = math.sqrt(npoints).toInt): Unit = {
    this.featureMap = (points: List[DenseVector[Double]]) => {
      points.map((p) =>
      {
        p :/= kernel.getBandwidth
        val n = norm(p, 2)
        DenseVector.tabulate[Double](M){i =>
          val lambda = math.pow(math.pow(2, i)*utils.factorial(i)*math.sqrt(math.Pi), 0.5)
          math.exp(-n*n)*utils.hermite(i, n)/lambda
        }
      }
      )
    }
    this.params = DenseVector.ones[Double](M + 1)
    this.applyFeatureMap()
  }

  /**
   * Override the effect of appyling a kernel
   * and return the model back to its default
   * state i.e. the Identity Kernel
   * */
  override def clearParameters(): Unit = {
    this.params = DenseVector.ones[Double](this.featuredims)
    this.featureMap = (x) => x
    val it = this.getParamOutEdges().iterator()
    while(it.hasNext) {
      val outEdge = it.next()
      val ynode = outEdge.getVertex(Direction.IN)
      val xnode = ynode.getEdges(Direction.IN,"causes")
        .iterator().next().getVertex(Direction.OUT)
      xnode.setProperty(
        "featureMap",
        xnode.getProperty("value")
          .asInstanceOf[DenseVector[Double]]
      )
      this.g.getVertex("w").setProperty("slope", this.params)
    }
  }

  //TODO: Replace stub implementations with the real ones
  override def crossvalidate(): Double = {
    0.0
  }

  override def tuneRBFKernel(): Unit = {
    //Generate a grid of sigma values
  }
}
