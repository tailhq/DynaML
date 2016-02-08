package io.github.mandar2812.dynaml.optimization

import breeze.linalg.DenseVector
import io.github.mandar2812.dynaml.graphutils.Neuron
import io.github.mandar2812.dynaml.models.neuralnets.FFNeuralGraph

/**
  * Created by mandar on 7/2/16.
  */
class BackPropogation extends RegularizedOptimizer[Int, FFNeuralGraph,
  DenseVector[Double], DenseVector[Double],
  Stream[(DenseVector[Double], DenseVector[Double])]] {
  /**
    * Solve the convex optimization problem.
    */
  override def optimize(nPoints: Long,
                        ParamOutEdges: Stream[(DenseVector[Double], DenseVector[Double])],
                        initialP: FFNeuralGraph): FFNeuralGraph = {
    (1 to this.numIterations).foreach{iteration =>
      ParamOutEdges.foreach(dataPattern => {
        //Perform Forward Pass
        initialP.forwardPass(dataPattern._1)
        //Set output node values
        initialP
          .getLayer(initialP.hidden_layers+1)
          .foreach(n => n.setValue(dataPattern._2(n.getNID()-1)))
        //Perform weight updates.
        (1 to initialP.hidden_layers+1).foreach(layer => {
          initialP.getLayerSynapses(layer).foreach(synapse => {
            val preSN = synapse.getPreSynapticNeuron()
            val postSN = synapse.getPostSynapticNeuron()
            //For each synapse perform weight update as
            // delta(w) = learning_rate*grad(postSN)*localfield(preSN)
            val origWeight = synapse.getWeight()
            synapse.setWeight(origWeight+(this.stepSize*
              Neuron.getLocalGradient(postSN, initialP.hidden_layers)*
              Neuron.getLocalField(preSN)._1))
          })
        })
      })
    }

    initialP
  }
}
