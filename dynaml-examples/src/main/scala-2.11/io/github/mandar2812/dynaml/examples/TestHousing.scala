package io.github.mandar2812.dynaml.examples

import com.tinkerpop.blueprints.Graph
import com.tinkerpop.frames.FramedGraph
import io.github.mandar2812.dynaml.graph.CausalEdge
import io.github.mandar2812.dynaml.models.KernelizedModel
import io.github.mandar2812.dynaml.models.svm.LSSVMModel

/**
 * @author mandar2812 on 22/6/15.
 */

object TestHousing {
  def apply(prototypes: Int = 1, kernel: String,
            globalOptMethod: String = "gs", grid: Int = 7,
            step: Double = 0.3, logscale: Boolean = false, csaIt: Int = 5): Unit = {
    val config = Map("file" -> "data/bostonhousing.csv", "delim" -> ",",
      "head" -> "false",
      "task" -> "regression")

    val configtest = Map("file" -> "data/bostonhousingtest.csv",
      "delim" -> ",",
      "head" -> "false")



    val model = LSSVMModel(config)

    val nProt = if (kernel == "Linear") {
      model.npoints.toInt
    } else {
      if(prototypes > 0)
        prototypes
      else
        math.sqrt(model.npoints.toDouble).toInt
    }

    val (optModel, optConfig) = KernelizedModel.getOptimizedModel[FramedGraph[Graph],
      Iterable[CausalEdge], model.type](model, globalOptMethod,
        kernel, nProt, grid, step, logscale, csaIt)

    optModel.learn()

    val met = optModel.evaluate(configtest)

    met.print()

    met.generatePlots()
  }
}

