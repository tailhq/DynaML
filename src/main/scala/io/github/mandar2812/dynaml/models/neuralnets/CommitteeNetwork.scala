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
package io.github.mandar2812.dynaml.models.neuralnets

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.models.LinearModel
import io.github.mandar2812.dynaml.optimization.{BackPropogation, CommitteeModelSolver, RegularizedOptimizer}
import io.github.mandar2812.dynaml.pipes.DataPipe

/**
  * Created by mandar on 9/2/16.
  */
class CommitteeNetwork[D](data: D,
                          transform: DataPipe[D, Stream[(DenseVector[Double],
                            DenseVector[Double])]],
                          networks: FFNeuralGraph*) extends
LinearModel[D, DenseVector[Double], DenseVector[Double],
  Double, Stream[(DenseVector[Double], Double)]] {

  override protected val g: D = data

  val num_points = dataAsStream(g).length

  val baseNetworks: List[FFNeuralGraph] = networks.toList

  val baseOptimizer = new BackPropogation()
    .setMomentum(0.01).setRegParam(0.001)
    .setMiniBatchFraction(1.0/baseNetworks.length)
    .setNumIterations(20)

  override protected val optimizer: RegularizedOptimizer[DenseVector[Double],
    DenseVector[Double], Double,
    Stream[(DenseVector[Double], Double)]] = new CommitteeModelSolver()

  override protected var params: DenseVector[Double] =
    DenseVector.fill[Double](baseNetworks.length)(1.0)

  def dataAsStream(d: D) = transform.run(d)

  /**
    * Predict the value of the
    * target variable given a
    * point.
    *
    **/
  override def predict(point: DenseVector[Double]): Double =
    params dot featureMap(point)

  override def clearParameters(): Unit =
    DenseVector.fill[Double](baseNetworks.length)(1.0)

  override def initParams(): DenseVector[Double] =
    DenseVector.fill[Double](baseNetworks.length)(1.0)

  featureMap = (pattern) =>
    DenseVector(baseNetworks.map(net => net.forwardPass(pattern)(0)).toArray)

  /**
    * Learn the parameters
    * of the model which
    * are in a node of the
    * graph.
    *
    **/
  override def learn(): Unit = {
    //First learn the base Networks
    baseNetworks.foreach(network => {
      baseOptimizer.optimize(num_points,dataAsStream(g),network)
    })

    params = optimizer.optimize(
      num_points,
      dataAsStream(g).map(couple =>
        (featureMap(couple._1), couple._2(0))),
      initParams()
    )
  }

  def setMomentum(m: Double): this.type = {
    this.baseOptimizer.setMomentum(m)
    this
  }


  def test(d: D): Stream[(DenseVector[Double], DenseVector[Double])] = {
    val (procInputs, _) =
      dataAsStream(d)
        .map(c =>
          (c._1.toArray.toList.map(i => List(i)), c._2.toArray.toList.map(i => List(i))))
        .reduce((c1,c2) =>
          (c1._1.zip(c2._1).map(c => c._1++c._2), c1._2.zip(c2._2).map(c => c._1++c._2)))


    val committeepredictions = baseNetworks.map(network => {
      network.predictBatch(procInputs)
    })

    dataAsStream(d).map(_._2).zipWithIndex.map(c => {
      val votes = DenseVector.tabulate[Double](baseNetworks.length)(Ndim =>
        committeepredictions(Ndim)(0)(c._2))

      val prediction: Double = params dot votes

      (DenseVector(prediction), c._1)
    })
  }
}
