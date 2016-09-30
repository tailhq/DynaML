package io.github.mandar2812.dynaml.kernels

import io.github.mandar2812.dynaml.algebra.SparkMatrix
import org.apache.spark.rdd.RDD

/**
  * Created by mandar on 28/09/2016.
  */
abstract class DistributedKernel[T] extends Kernel[T, Double] {

  def buildKernelMatrix(data: RDD[(Long, T)]) =
    new SparkMatrix(data.cartesian(data).map(c => ((c._1._1, c._2._1), evaluate(c._1._2, c._2._2))))

  def buildCrossKernelMatrix(data1: RDD[(Long, T)], data2: RDD[(Long, T)]) =
    new SparkMatrix(data1.cartesian(data2).map(c => ((c._1._1, c._2._1), evaluate(c._1._2, c._2._2))))

}
