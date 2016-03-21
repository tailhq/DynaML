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
import io.github.mandar2812.dynaml.graphutils.Neuron
import io.github.mandar2812.dynaml.models.neuralnets.FFNeuralGraph
import org.apache.log4j.Logger

import scala.util.Random

/**
  * Implementation of the standard back pro-pogation with momentum
  * using the "generalized delta rule".
  */
class BackPropogation extends RegularizedOptimizer[Int, FFNeuralGraph,
  DenseVector[Double], DenseVector[Double],
  Stream[(DenseVector[Double], DenseVector[Double])]] {

  private val logger = Logger.getLogger(this.getClass)

  protected var momentum: Double = 0.0

  def setMomentum(m: Double): this.type = {
    momentum = m
    this
  }

  /**
    * Solve the convex optimization problem.
    */
  override def optimize(nPoints: Long,
                        ParamOutEdges: Stream[(DenseVector[Double], DenseVector[Double])],
                        initialP: FFNeuralGraph): FFNeuralGraph = {
    logger.info(" Configuration ")
    logger.info("---------------")
    logger.info(" Mini Batch Fraction : "+miniBatchFraction)
    logger.info(" Max Iterations : "+numIterations)
    logger.info(" Learning Rate : "+stepSize)
    (1 to this.numIterations).foreach{iteration =>
      val (procInputs, procOutputs) =
        ParamOutEdges
          .filter(_ => Random.nextDouble() <= miniBatchFraction)
          .map(c =>
            (c._1.toArray.toList.map(i => List(i)), c._2.toArray.toList.map(i => List(i))))
          .reduce((c1,c2) =>
            (c1._1.zip(c2._1).map(c => c._1++c._2), c1._2.zip(c2._2).map(c => c._1++c._2)))

      logger.info(" ************** Iteration: "+iteration+" ************** ")
      logger.info(" Forward Pass ")
      //forward pass, set inputs
      (0 to initialP.hidden_layers+1).foreach(layer => {

        if(layer == 0) {

          initialP.getLayer(0).foreach(node => node.getNeuronType() match {
            case "input" =>
              node.setValueBuffer(procInputs(node.getNID() - 1).toArray)
              node.setLocalFieldBuffer(procInputs(node.getNID() - 1).toArray)
            case "bias" =>
              node.setValueBuffer(Array.fill[Double](procInputs.head.length)(1.0))
              node.setLocalFieldBuffer(Array.fill[Double](procInputs.head.length)(1.0))
          })

        } else if(layer == initialP.hidden_layers+1) {
          initialP.getLayer(initialP.hidden_layers+1).foreach(node =>{
            node.setValueBuffer(procOutputs(node.getNID() - 1).toArray)
            val (locfield, _) = Neuron.getLocalFieldBuffer(node)
            node.setLocalFieldBuffer(locfield)
            node.setLocalGradBuffer(Neuron.getLocalGradientBuffer(node, initialP.hidden_layers))
          })

        } else {
          initialP.getLayer(layer).foreach(node => node.getNeuronType() match {
            case "perceptron" =>
              val (locfield, field) = Neuron.getLocalFieldBuffer(node)
              node.setLocalFieldBuffer(locfield)
              node.setValueBuffer(field)
            case "bias" =>
              node.setValueBuffer(Array.fill[Double](procInputs.head.length)(1.0))
              node.setLocalFieldBuffer(Array.fill[Double](procInputs.head.length)(1.0))
          })
        }
      })

      //Backward pass calculate local gradients
      logger.info(" Backward Pass ")
      (1 to initialP.hidden_layers+1).reverse.foreach{layer => {

        initialP.getLayer(layer).foreach(node => {
          //logger.info("Backward Pass Layer: "+layer + "Node "+node.getNID())
          node.setLocalGradBuffer(Neuron.getLocalGradientBuffer(node, initialP.hidden_layers))
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
          val regularizationTerm = regParam*origWeight

          val weightUpdate = this.stepSize*postG.zip(preF).map(c => c._1*c._2).sum/(nPoints*miniBatchFraction) +
            momentumTerm + regularizationTerm

          synapse.setWeight(origWeight + weightUpdate)
          synapse.setPrevWeightUpdate(weightUpdate)
        })
      })
    }

    initialP
  }
}
