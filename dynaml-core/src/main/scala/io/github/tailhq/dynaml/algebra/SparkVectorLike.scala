package io.github.tailhq.dynaml.algebra

import org.apache.spark.rdd.RDD

/**
  * @author tailhq date: 13/10/2016.
  */
trait SparkVectorLike[V] {

  protected var vector: RDD[(Long, V)]

  def _vector = vector

  protected def vector_(other: RDD[(Long, V)]): Unit = {
    vector = other
  }

  def <~(other: SparkVectorLike[V]): Unit = {
    vector_(other._vector)
  }

}
