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
import io.github.mandar2812.dynaml.DynaMLPipe._
import io.github.mandar2812.dynaml.graph.FFNeuralGraph
import io.github.mandar2812.dynaml.graph.utils.Neuron._
import io.github.mandar2812.dynaml.pipes.{DataPipe, StreamDataPipe}
import org.apache.log4j.Logger
import spire.implicits._
import scala.util.Random

/**
  * Implementation of the standard back pro-pogation with momentum
  * using the "generalized delta rule".
  */
class BackPropagation extends RegularizedOptimizer[FFNeuralGraph,
  DenseVector[Double], DenseVector[Double],
  Stream[(DenseVector[Double], DenseVector[Double])]] {

  protected var momentum: Double = 0.0

  protected var sparsityWeight: Double = 0.0

  def setMomentum(m: Double): this.type = {
    momentum = m
    this
  }

  def setSparsityWeight(s: Double): this.type = {
    sparsityWeight = s
    this
  }

  /**
    * Solve the convex optimization problem.
    */
  override def optimize(nPoints: Long,
                        ParamOutEdges: Stream[(DenseVector[Double], DenseVector[Double])],
                        initialP: FFNeuralGraph): FFNeuralGraph = BackPropagation.run(
    nPoints, this.regParam, this.numIterations, this.miniBatchFraction,
    this.stepSize, this.momentum, initialP, ParamOutEdges,
    identityPipe[Stream[(DenseVector[Double], DenseVector[Double])]],
    sparsityWeight
  )
}

object BackPropagation {

  val logger = Logger.getLogger(this.getClass)

  var rho = 0.05

  /**
    * Processes the raw data into neuron buffers,
    * represented as a DynaML pipe
    * */
  val processDataToNeuronBuffers =
    DataPipe((data: Stream[(DenseVector[Double], DenseVector[Double])]) =>
      data.map(c =>
        (c._1.toArray.toList.map(i => List(i)), c._2.toArray.toList.map(i => List(i)))
      ).reduce((c1,c2) =>
        (c1._1.zip(c2._1).map(c => c._1++c._2), c1._2.zip(c2._2).map(c => c._1++c._2))
      )
    )

  /**
    * Carry out the iterative backpropagation algorithm to
    * determine synapse weights in a feed-forward neural network graph
    * */
  def run[T](nPoints: Long, regParam: Double, numIterations: Int,
             miniBatchFraction: Double, stepSize: Double, momentum: Double,
             initialP: FFNeuralGraph, ParamOutEdges: T,
             transform: DataPipe[T, Stream[(DenseVector[Double], DenseVector[Double])]],
             sparsityWeight: Double = 0.0) = {

    //log important back-propagation parameters to the screen
    logger.info(" Configuration ")
    logger.info("---------------")
    logger.info(" Mini Batch Fraction : "+miniBatchFraction)
    logger.info(" Max Iterations : "+numIterations)
    logger.info(" Learning Rate : "+stepSize)
    logger.info(" Regularization : "+regParam)
    logger.info(" Momentum: "+momentum)

    //Calculate the effective data size based on the
    //mini batch fraction and the total number of
    //training data points
    val effectiveDataSize = nPoints*miniBatchFraction

    val dataToBuffersPipe = transform >
      StreamDataPipe((pattern: (DenseVector[Double], DenseVector[Double])) =>
        Random.nextDouble() <= miniBatchFraction) >
      processDataToNeuronBuffers

    val (procInputs, procOutputs) = dataToBuffersPipe(ParamOutEdges)

    //Initialize bias units
    (1 to initialP.hidden_layers).foreach(layer => {
      initialP.getLayer(layer).filter(_.getNeuronType() == "bias").foreach(node => {
        node.setValueBuffer(Array.fill[Double](procInputs.head.length)(1.0))
        node.setLocalFieldBuffer(Array.fill[Double](procInputs.head.length)(1.0))
      })
    })

    //Fill input layer with features from training data
    initialP.getLayer(0).foreach(node => node.getNeuronType() match {
      case "input" =>
        node.setValueBuffer(procInputs(node.getNID() - 1).toArray)
        node.setLocalFieldBuffer(procInputs(node.getNID() - 1).toArray)
      case "bias" =>
        node.setValueBuffer(Array.fill[Double](procInputs.head.length)(1.0))
        node.setLocalFieldBuffer(Array.fill[Double](procInputs.head.length)(1.0))
    })

    //Fill output layer with target values from training data
    initialP.getLayer(initialP.hidden_layers+1).foreach(node =>{
      node.setValueBuffer(procOutputs(node.getNID() - 1).toArray)
    })

    //Begin back-propagation iterations

    cfor(1)(iteration => iteration < numIterations, iteration => iteration + 1)( iteration => {

      val damping = stepSize/(1+0.5*iteration)
      logger.info(" ************** Iteration: "+iteration+" ************** ")
      logger.info(" Forward Pass ")
      //forward pass, set inputs
      (1 to initialP.hidden_layers+1).foreach(layer => {

        if(layer == initialP.hidden_layers+1) {
          initialP.getLayer(initialP.hidden_layers+1).foreach(node =>{
            val (locfield, _) = getLocalFieldBuffer(node)
            node.setLocalFieldBuffer(locfield)

            //Set gradient values for output node
            node.setLocalGradBuffer(
              getLocalGradientBuffer(
                node, initialP.hidden_layers,
                rho, sparsityWeight)
            )
          })

        } else {
          initialP.getLayer(layer)
            .filter(_.getNeuronType() == "perceptron")
            .foreach(node => {
              val (locfield, field) = getLocalFieldBuffer(node)
              node.setLocalFieldBuffer(locfield)
              node.setValueBuffer(field)
            })
        }
      })

      //Backward pass calculate local gradients
      logger.info(" Backward Pass ")
      (1 to initialP.hidden_layers).reverse.foreach{layer => {

        initialP.getLayer(layer)
          .filter(_.getNeuronType() == "perceptron")
          .foreach(node => {
            node.setLocalGradBuffer(
              getLocalGradientBuffer(node, initialP.hidden_layers, rho, sparsityWeight)
            )
          })
      }}

      //Recalculate weights
      logger.info(" Weight Update ")
      (1 to initialP.hidden_layers+1).reverse.foreach(layer => {
        initialP.getLayerSynapses(layer).foreach(synapse => {
          val preSN = synapse.getPreSynapticNeuron()
          val postSN = synapse.getPostSynapticNeuron()

          //For each synapse perform weight update as
          // delta(w) = learning_rate*grad(postSN)*localfield(preSN)

          val origWeight = synapse.getWeight()
          val postG = postSN.getLocalGradBuffer()
          val preF = preSN.getLocalFieldBuffer()

          val momentumTerm = momentum*synapse.getPrevWeightUpdate()

          //Calculate the net gradient due to all data points at the particular synapse
          val (netGradientContribution, regularizationTerm) =
            preSN.getNeuronType() match {
              case "bias" =>
                (postG.sum, 0.0)
              case _ =>
                (postG.zip(preF).map(c => c._1*c._2).sum, regParam*origWeight)
            }

          //Calculate the synapse weight update
          val weightUpdate = damping*netGradientContribution/effectiveDataSize +
            momentumTerm +
            regularizationTerm

          //Update the synapse weight
          synapse.setWeight(origWeight - weightUpdate)
          synapse.setPrevWeightUpdate(weightUpdate)
        })
      })
    })
    initialP
  }
}