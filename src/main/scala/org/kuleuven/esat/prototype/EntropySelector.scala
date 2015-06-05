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

package org.kuleuven.esat.prototype

import breeze.linalg.DenseVector
import org.apache.log4j.{Priority, Logger}
import org.kuleuven.esat.graphicalModels.KernelBayesianModel

/**
 * Basic skeleton of an entropy based
 * subset selector
 */
abstract class EntropySelector
  extends SubsetSelector[KernelBayesianModel, DenseVector[Double]]
  with Serializable {
  protected val measure: EntropyMeasure
  protected val delta: Double
  protected val MAX_ITERATIONS: Int
}

private[esat] class GreedyEntropySelector(
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
      data: KernelBayesianModel,
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

  def subsetSelection(data: KernelBayesianModel,
                      M: Int,
                      measure: EntropyMeasure, delta: Double,
                      MAX_ITERATIONS: Int): List[Int] = {

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
    logger.log(Priority.INFO, "Initializing the working set, by drawing randomly from the training set")
    var workingset = r.shuffle(1 to data.npoints).toList.slice(0, M)

    //All the elements not in the working set
    var newDataset = (1 to data.npoints).filter((p) => !workingset.contains(p))
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
      } else {
        /*
        * No improvement in entropy
        * so revert the working set
        * to its initial state. Leave
        * the variable newDataset as
        * it is.
        * */
        workingset = (workingset :+ point1).filter((p) => p != point2)
      }

      logger.info("Iteration: "+it)
    } while(math.abs(d/oldEntropy) >= delta &&
      it <= MAX_ITERATIONS)
    logger.log(Priority.INFO, "Returning final prototype set")
    //Time to return the final working set
    workingset
  }
}
