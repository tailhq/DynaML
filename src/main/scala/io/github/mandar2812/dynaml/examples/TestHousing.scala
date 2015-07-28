package io.github.mandar2812.dynaml.examples

import com.tinkerpop.blueprints.Graph
import com.tinkerpop.frames.FramedGraph
import io.github.mandar2812.dynaml.graphutils.CausalEdge
import io.github.mandar2812.dynaml.models.{KernelizedModel, GaussianLinearModel}
import io.github.mandar2812.dynaml.optimization.GridSearch
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * @author mandar2812 on 22/6/15.
 */

object TestHousing {
  def apply(prototypes: Int = 1, kernel: String,
            globalOptMethod: String = "gs", grid: Int = 7,
            step: Double = 0.3, logscale: Boolean = false): Unit = {
    val config = Map("file" -> "data/bostonhousing.csv", "delim" -> ",",
      "head" -> "false",
      "task" -> "regression")

    val configtest = Map("file" -> "data/bostonhousingtest.csv",
      "delim" -> ",",
      "head" -> "false")



    val model = GaussianLinearModel(config)

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
        kernel, nProt, grid, step, logscale)

    optModel.setMaxIterations(40).learn()

    val met = optModel.evaluate(configtest)

    met.print()
  }
}

