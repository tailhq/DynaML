package io.github.tailhq.dynaml.kernels

import io.github.tailhq.dynaml.algebra.{SparkMatrix, SparkPSDMatrix}
import org.apache.spark.rdd.RDD

/**
  * Created by mandar on 28/09/2016.
  */
trait DistributedKernel[T] extends Kernel[T, Double] {

  def buildKernelMatrix(data: RDD[(Long, T)]) = SparkPSDMatrix(data)(this.evaluate)

  def buildCrossKernelMatrix(data1: RDD[(Long, T)], data2: RDD[(Long, T)]) = SparkMatrix(data1, data2)(this)

}
