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
import io.github.mandar2812.dynaml.graph.FFNeuralGraph
import io.github.mandar2812.dynaml.graph.utils.Neuron
import io.github.mandar2812.dynaml.optimization.BackPropagation
import io.github.mandar2812.dynaml.pipes.{ReversibleScaler, Scaler}

/**
  * @author mandar2812 22/6/16.
  *
  * Base implementation of a Sparse Autoencoder
  *
  * It is represented as a [[ReversibleScaler]] transforming
  * a breeze [[DenseVector]] into another breeze Dense Vector.
  */
class AutoEncoder(inDim: Int, outDim: Int,
                  acts: List[String] = List("logsig", "logsig"))
  extends ReversibleScaler[DenseVector[Double]]{

  def initialize() =
    FFNeuralGraph(
      inDim, inDim, 1, acts,
      List(outDim))

  var graph = initialize()

  val optimizer = new BackPropagation

  val i = Scaler((xhat: DenseVector[Double]) => {
    graph.getLayer(1)
      .filter(_.getNeuronType() == "perceptron")
      .foreach(n => n.setValue(xhat(n.getNID()-1)))

    val outputs:Map[Int, Double] = graph.getLayer(2).filter(_.getNeuronType() == "perceptron")
      .map(outputNeuron => (outputNeuron.getNID(), Neuron.getLocalField(outputNeuron)._1))
      .toMap

    DenseVector.tabulate[Double](inDim)(i => outputs(i+1))
  })

  def learn(data: Stream[(DenseVector[Double], DenseVector[Double])]) = {
    graph = optimizer.optimize(data.length.toLong, data, initialize())
  }

  override def run(x: DenseVector[Double]) = {
    graph.forwardPass(x)

    val outputs:Map[Int, Double] = graph.getLayer(1).filter(_.getNeuronType() == "perceptron")
      .map(outputNeuron => (outputNeuron.getNID(), Neuron.getLocalField(outputNeuron)._1))
      .toMap

    DenseVector.tabulate[Double](outDim)(i => outputs(i+1))

  }

}
