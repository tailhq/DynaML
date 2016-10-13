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

import scala.collection.immutable.NumericRange

/**
  * @author mandar2812 date: 28/09/2016.
  *
  * A row vector backed by a spark [[RDD]]
  */
class DualSparkVector(baseDualVector: RDD[(Long, Double)])
  extends SparkMatrix(baseDualVector.map(c => ((0L, c._1), c._2)))
    with NumericOps[DualSparkVector] with SparkVectorLike[Double] {

  override lazy val rows = 1L

  override def repr: DualSparkVector = this

  protected var vector = baseDualVector

  override def t: SparkVector = new SparkVector(_vector)

  def apply(r: NumericRange[Long]): DualSparkVector =
    new DualSparkVector(_vector.filterByRange(r.min, r.max).map(e => (e._1-r.min, e._2)))

  def apply(r: Range): DualSparkVector =
    new DualSparkVector(_vector.filterByRange(r.min.toLong, r.max.toLong).map(e => (e._1-r.min, e._2)))

}

object DualSparkVector {

  /**
    * Tabulate a [[DualSparkVector]]
    */
  def apply(list: RDD[Long])(eval: (Long) => Double) = new DualSparkVector(list.map(e => (e, eval(e))))


  def horzcat(vectors: DualSparkVector*): DualSparkVector = {
    val sizes = vectors.map(_.cols)
    new DualSparkVector(vectors.zipWithIndex.map(couple => {
      val offset = sizes.slice(0, couple._2).sum
      couple._1._vector.map(c => (c._1+offset, c._2))
    }).reduce((a,b) => a.union(b)))
  }

}