package io.github.mandar2812.dynaml

import io.github.mandar2812.dynaml.pipes.DataPipe
import org.apache.spark.{SparkConf, SparkContext}

/**
  * @author mandar date 24/01/2017.
  *
  * Pipelines for initializing Apache Spark
  */
object DynaMLSpark {

  val sparkConfigPipe = DataPipe((appAndHost: (String, String)) => {
    new SparkConf().setMaster(appAndHost._2).setAppName(appAndHost._1)
  })

  val sparkContextPipe = DataPipe((conf: SparkConf) => new SparkContext(conf))

  val initializeSpark = sparkConfigPipe > sparkContextPipe
}
