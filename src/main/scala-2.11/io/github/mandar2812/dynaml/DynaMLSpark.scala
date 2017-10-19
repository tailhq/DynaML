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
