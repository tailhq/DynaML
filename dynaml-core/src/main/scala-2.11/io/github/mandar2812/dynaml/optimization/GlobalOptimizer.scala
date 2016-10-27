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
package io.github.mandar2812.dynaml.optimization

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.utils
import org.apache.log4j.Logger

/**
 * @author mandar2812 datum 24/6/15.
 *
 * High level interface defining the
 * core functions of a global optimizer
 */
trait GlobalOptimizer[T <: GloballyOptimizable] {

  val system: T

  protected val logger = Logger.getLogger(this.getClass)

  protected var step: Double = 0.3

  protected var gridsize: Int = 3

  protected var logarithmicScale = false

  def setLogScale(t: Boolean) = {
    logarithmicScale = t
    this
  }

  def setGridSize(s: Int) = {
    this.gridsize = s
    this
  }

  def setStepSize(s: Double) = {
    this.step = s
    this
  }

  def getEnergyLandscape(initialConfig: Map[String, Double],
                         options: Map[String, String] = Map())
  : List[(Double, Map[String, Double])] = {

    //create grid

    //one list for each key in initialConfig
    val hyper_params = initialConfig.keys.toList

    def scaleFunc(param: String) = if(logarithmicScale)
      (i: Int) => {initialConfig(param)/math.exp((i+1).toDouble*step)}
    else
      (i: Int) => initialConfig(param) - (i+1).toDouble*step

    val gridvecs = initialConfig.map((keyValue) => {
      (keyValue._1, List.tabulate(gridsize)(scaleFunc(keyValue._1)))
    })

    val grid = utils.combine(gridvecs.values).map(x => DenseVector(x.toArray))

    grid.map((config) => {
      val configMap = List.tabulate(config.length){i => (hyper_params(i), config(i))}.toMap
      logger.info("""Evaluating Configuration: """+GlobalOptimizer.prettyPrint(configMap))

      val configEnergy = system.energy(configMap, options)

      logger.info("Energy = "+configEnergy+"\n")

      (configEnergy, configMap)
    }).toList

  }

  def optimize(initialConfig: Map[String, Double],
               options: Map[String, String] = Map()): (T, Map[String, Double])

}

object GlobalOptimizer {

  def prettyPrint(configuration: Map[String, Double]): String = configuration.foldLeft("""""")(
    (str, mapping) => str+""" """+mapping._1+""" = """+"%4f".format(mapping._2).toString)

}
