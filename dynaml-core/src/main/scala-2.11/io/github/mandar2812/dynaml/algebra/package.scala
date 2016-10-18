package io.github.mandar2812.dynaml

import breeze.linalg.scaleAdd

/**
  * Created by mandar on 13/10/2016.
  */
package object algebra {

  def axpyDist[X <: SparkMatrix, Y <: SparkMatrix](a: Double, x: X, y: Y)(
    implicit axpy: scaleAdd.InPlaceImpl3[Y, Double, X]): Unit = {
    axpy(y, a, x)
  }

  def axpyDist(a: Double, x: SparkBlockedVector, y: SparkBlockedVector)(
    implicit axpy: scaleAdd.InPlaceImpl3[SparkBlockedVector, Double, SparkBlockedVector]): Unit = {
    axpy(y, a, x)
  }

  def axpyDist(a: Double, x: PartitionedVector, y: PartitionedVector)(
    implicit axpy: scaleAdd.InPlaceImpl3[PartitionedVector, Double, PartitionedVector]): Unit = {
    axpy(y, a, x)
  }


}
