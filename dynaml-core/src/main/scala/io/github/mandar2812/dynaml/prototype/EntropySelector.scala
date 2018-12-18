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
package io.github.mandar2812.dynaml.prototype

import breeze.linalg.DenseVector
import org.apache.log4j.{Priority, Logger}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer


object GreedyEntropySelector {
  private val logger = Logger.getLogger(this.getClass)


  /**
   * Implementation of Quadratic Renyi Entropy
   * based iterative subset selection. The function
   * is written as a special case for Big Data Problems
   * because of the expression for Quadratic Renyi Entropy
   * allows for O(n) time update instead of O(n**2)
   *
   * */
  def subsetSelectionQRE(data: RDD[(Long, LabeledPoint)],
                      measure: QuadraticRenyiEntropy, M: Int,
                      MAX_ITERATIONS: Int,
                      delta: Double): List[DenseVector[Double]] = {

    /*
    * Draw an initial sample of M points
    * from data without replacement.
    *
    * Define a working set which we
    * will use as a prototype set to
    * to each iteration
    * */
    logger.info("Initializing the working set, by drawing randomly from the training set")

    val workingset = ArrayBuffer(data.takeSample(false, M):_*)
    val workingsetIndices = workingset.map(_._1)

    val r = scala.util.Random
    var it: Int = 0

    // All the elements not in the working set
    var newDataset = data.filter((p) => !workingsetIndices.contains(p._1))
    // Existing best value of the entropy
    var oldEntropy: Double = measure.evaluate(workingset.map(_._2.features.toArray)
      .map(DenseVector(_)).toList)
    var newEntropy = oldEntropy
    var d: Double = Double.NegativeInfinity
    var last_pos_d = Double.PositiveInfinity
    var rand: Int = 0
    logger.info("Starting iterative, entropy based greedy subset selection")
    do {
      /*
       * Randomly select a point from
       * the working set as well as data
       * and then swap them.
       * */
      rand = r.nextInt(workingset.length - 1)
      val point1 = workingset(rand)

      val point2 = newDataset.takeSample(false, 1).apply(0)

      workingset -= point1
      workingset += point2
      workingsetIndices -= point1._1
      workingsetIndices += point2._1

      /*newEntropy = measure.evaluate(workingset.map(p =>
        DenseVector(p._2.features.toArray)).toList)*/

      /*
      * Calculate the change in entropy,
      * if it has improved then keep the
      * swap, otherwise revert to existing
      * working set.
      * */
      /*newEntropy - oldEntropy*/
      d = measure.entropyDifference(oldEntropy, workingset.map(_._2.features.toArray)
        .map(DenseVector(_)).toList, DenseVector(point2._2.features.toArray),
        DenseVector(workingset(rand)._2.features.toArray))

      it += 1
      if(d > 0) {
        /*
        * Improvement in entropy so
        * keep the updated working set
        * as it is and update the
        * variable 'newDataset'
        * */

        oldEntropy += d
        last_pos_d = d
        newDataset = data.filter((p) => !workingsetIndices.contains(p._1))
      } else {
        workingset += point1
        workingset -= point2
        workingsetIndices += point1._1
        workingsetIndices -= point2._1
      }

    } while(last_pos_d >= delta &&
      it <= MAX_ITERATIONS)
    logger.info("Working set obtained, now starting process of packaging it as an RDD")
    // Time to return the final working set
    workingset.map(i => DenseVector(i._2.features.toArray)).toList
  }


}
