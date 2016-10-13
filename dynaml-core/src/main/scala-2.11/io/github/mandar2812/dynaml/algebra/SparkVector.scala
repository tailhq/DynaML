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

import breeze.linalg.NumericOps
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.immutable.NumericRange

/**
  * @author mandar2812 date: 28/09/2016.
  *
  * A distributed vector backed by a spark [[RDD]]
  */
class SparkVector(baseVector: RDD[(Long, Double)], size: Long = -1L, sanityChecks: Boolean = true)
  extends SparkMatrix(baseVector.map(c => ((c._1, 0L), c._2)), size, 1L, sanityChecks)
    with NumericOps[SparkVector] with SparkVectorLike[Double] {

  protected var vector = baseVector

  override lazy val cols = 1L

  override def repr: SparkVector = this

  override def t: DualSparkVector = new DualSparkVector(_vector)

  def apply(r: NumericRange[Long]): SparkVector =
    new SparkVector(_vector.filterByRange(r.min, r.max).map(e => (e._1-r.min, e._2)))

  def apply(r: Range): SparkVector =
    new SparkVector(_vector.filterByRange(r.min, r.max).map(e => (e._1-r.min, e._2)))

  override def persist: Unit = {
    vector.persist(StorageLevel.MEMORY_AND_DISK)
  }

  override def unpersist: Unit = {
    vector.unpersist()
  }

}


object SparkVector {

  /**
    * Tabulate a [[SparkVector]]
    */
  def apply(list: RDD[Long])(eval: (Long) => Double) = new SparkVector(list.map(e => (e, eval(e))))


  def vertcat(vectors: SparkVector*): SparkVector = {
    val sizes = vectors.map(_.rows)
    new SparkVector(vectors.zipWithIndex.map(couple => {
      val offset = sizes.slice(0, couple._2).sum
      couple._1._vector.map(c => (c._1+offset, c._2))
    }).reduce((a,b) => a.union(b)))
  }

}