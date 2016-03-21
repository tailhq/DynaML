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
import io.github.mandar2812.dynaml.models.svm.KernelLSSVMModel
import org.apache.log4j.{Priority, Logger}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
 * Basic skeleton of an entropy based
 * subset selector
 */
abstract class EntropySelector
  extends SubsetSelector[KernelLSSVMModel, DenseVector[Double]]
  with Serializable {
  protected val measure: EntropyMeasure
  protected val delta: Double
  protected val MAX_ITERATIONS: Int
}

private[dynaml] class GreedyEntropySelector(
    m: EntropyMeasure,
    del: Double = 0.0001,
    max: Int = 1000)
  extends EntropySelector
  with Serializable {

  override protected val measure: EntropyMeasure = m
  override protected val delta: Double = del
  override protected val MAX_ITERATIONS: Int =  max
  private val logger = Logger.getLogger(this.getClass)

  override def selectPrototypes(
      data: KernelLSSVMModel,
      M: Int): List[DenseVector[Double]] = {

    val prototypeIndexes =
      GreedyEntropySelector.subsetSelection(
        data,
        M,
        this.measure,
        this.delta,
        this.MAX_ITERATIONS
      )

    data.filter((p) =>
      prototypeIndexes.contains(p)
    )
  }
}

object GreedyEntropySelector {
  private val logger = Logger.getLogger(this.getClass)

  def apply(
    m: EntropyMeasure,
    del: Double = 0.0001,
    max: Int = 5000): GreedyEntropySelector =
    new GreedyEntropySelector(m, del, max)

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

  def subsetSelection(data: KernelLSSVMModel,
                      M: Int,
                      measure: EntropyMeasure, delta: Double,
                      MAX_ITERATIONS: Int): List[Long] = {

    /*
    * Draw an initial sample of M points
    * from data without replacement.
    *
    * Define a working set which we
    * will use as a prototype set to
    * to each iteration
    * */


    val r = scala.util.Random
    var it: Int = 0
    var it2: Int = 0
    logger.log(Priority.INFO, "Initializing the working set, by drawing randomly from the training set")
    var workingset = r.shuffle[Long, IndexedSeq](1L to data.npoints).toList.slice(0, M)

    //All the elements not in the working set
    var newDataset = (1L to data.npoints).filter((p) => !workingset.contains(p))
    //Existing best value of the entropy
    var oldEntropy: Double = measure.evaluate(data.filter((point) =>
      workingset.contains(point)))
    //Store the value of entropy after an element swap
    var newEntropy: Double = 0.0
    var d: Double = Double.NegativeInfinity
    var rand: Int = 0
    logger.log(Priority.INFO, "Starting iterative, entropy based greedy subset selection")
    do {
      /*
       * Randomly select a point from
       * the working set as well as data
       * and then swap them.
       * */
      rand = r.nextInt(workingset.length - 1)
      val point1 = r.shuffle(workingset).head

      val point2 = r.shuffle(newDataset).head

      //Update the working set
      workingset = (workingset :+ point2).filter((p) => p != point1)

      //Calculate the new entropy
      newEntropy = measure.evaluate(data.filter((p) =>
        workingset.contains(p)))

      /*
      * Calculate the change in entropy,
      * if it has improved then keep the
      * swap, otherwise revert to existing
      * working set.
      * */
      d = newEntropy - oldEntropy

      if(d > 0) {
        /*
        * Improvement in entropy so
        * keep the updated working set
        * as it is and update the
        * variable 'newDataset'
        * */
        oldEntropy = newEntropy
        newDataset = (newDataset :+ point1).filter((p) => p != point2)
        it += 1
        it2 = 0
      } else {
        /*
        * No improvement in entropy
        * so revert the working set
        * to its initial state. Leave
        * the variable newDataset as
        * it is.
        * */
        workingset = (workingset :+ point1).filter((p) => p != point2)
        it2 += 1
      }

      logger.info("Iteration: "+it)
    } while(math.abs(d/oldEntropy) >= delta &&
      it <= MAX_ITERATIONS && it2 <= MAX_ITERATIONS)
    logger.log(Priority.INFO, "Returning final prototype set")
    //Time to return the final working set
    workingset
  }
}
