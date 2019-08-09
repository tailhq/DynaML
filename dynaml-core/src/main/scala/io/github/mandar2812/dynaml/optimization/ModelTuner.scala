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
import breeze.stats.distributions.ContinuousDistr
import io.github.mandar2812.dynaml.probability.{
  ContinuousRVWithDistr,
  RandomVariable
}
import io.github.mandar2812.dynaml.utils
import org.apache.log4j.Logger

/**
  * A model tuner takes a model which implements [[GloballyOptimizable]] and
  * "tunes" it, returning (possibly) a model of a different type.
  * */
trait ModelTuner[T <: GloballyOptimizable, T1] {

  val system: T

  protected val logger: Logger = Logger.getLogger(this.getClass)

  protected var step: Double = 0.3

  protected var gridsize: Int = 3

  protected var logarithmicScale = false

  protected var num_samples: Int = 20

  protected var meanFieldPrior
    : Map[String, ContinuousRVWithDistr[Double, ContinuousDistr[Double]]] =
    Map()

  def setPrior(
    p: Map[String, ContinuousRVWithDistr[Double, ContinuousDistr[Double]]]
  ): this.type = {
    meanFieldPrior = p
    this
  }

  def setNumSamples(n: Int): this.type = {
    num_samples = n
    this
  }

  def setLogScale(t: Boolean): this.type = {
    logarithmicScale = t
    this
  }

  def setGridSize(s: Int): this.type = {
    this.gridsize = s
    this
  }

  def setStepSize(s: Double): this.type = {
    this.step = s
    this
  }

  def getGrid(initialConfig: Map[String, Double]): Seq[Map[String, Double]] = {

    val hyper_params = initialConfig.keys.toList

    def scaleFunc(param: String) =
      if (logarithmicScale) (i: Int) => {
        initialConfig(param) / math.exp((i + 1).toDouble * step)
      } else (i: Int) => initialConfig(param) - (i + 1).toDouble * step

    //one list for each key in initialConfig
    val gridvecs = initialConfig.map(keyValue => {
      (keyValue._1, List.tabulate(gridsize)(scaleFunc(keyValue._1)))
    })

    utils
      .combine(gridvecs.values)
      .map(x => DenseVector(x.toArray))
      .map(config => {
        List
          .tabulate(config.length) { i =>
            (hyper_params(i), config(i))
          }
          .toMap
      })
  }

  def getEnergyLandscape(
    initialConfig: Map[String, Double],
    options: Map[String, String] = Map(),
    prior: Map[String, ContinuousRVWithDistr[Double, ContinuousDistr[Double]]] =
      Map()
  ): List[(Double, Map[String, Double])] = {

    //create grid
    val hyp = initialConfig.keys

    val usePriorFlag: Boolean = hyp.forall(prior.contains)

    val priorRVAsMap =
      if (usePriorFlag) {
        RandomVariable(() => {
          prior.map(kv => (kv._1, kv._2.sample()))
        })
      } else {
        RandomVariable(() => initialConfig)
      }

    val grid: Seq[Map[String, Double]] =
      if (usePriorFlag) priorRVAsMap.iid(num_samples).sample()
      else getGrid(initialConfig)

    val pb = new utils.ProgressBar(grid.length)
    grid
      .map(config => {

        val configMap = config
        //println("\nEvaluating Configuration: ")
        //pprint.pprintln(configMap)

        val configEnergy = system.energy(configMap, options)

        val priorEnergy =
          if (usePriorFlag)
            configMap.foldLeft(0.0)(
              (p_acc, keyValue) =>
                p_acc - prior(keyValue._1).underlyingDist.logPdf(keyValue._2)
            )
          else 0.0

        val netEnergy = priorEnergy + configEnergy

        //print("Energy = ")
        //pprint.pprintln(configEnergy)

        /* if(usePriorFlag) {
        print("Energy due to Prior = ")
        pprint.pprintln(priorEnergy)
        print("Net Energy = ")
        pprint.pprintln(netEnergy)
      } */
        pb += 1
        (netEnergy, configMap)
      })
      .toList

  }

  def optimize(
    initialConfig: Map[String, Double],
    options: Map[String, String] = Map()
  ): (T1, Map[String, Double])

}
