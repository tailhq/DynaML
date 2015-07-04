import java.io.File
import com.github.tototoshi.csv.CSVWriter
import com.tinkerpop.blueprints.Graph
import com.tinkerpop.frames.FramedGraph
import org.kuleuven.esat.graphUtils.CausalEdge
import org.kuleuven.esat.models.{KernelizedModel, GaussianLinearModel}

/**
 * @author mandar2812
 */
object TestRipley {
  def apply(prototypes: Int = 1, kernel: String,
            globalOptMethod: String = "gs", grid: Int = 7,
            step: Double = 0.3, logscale: Boolean = false): Unit = {
    val config = Map("file" -> "data/ripley.csv", "delim" -> ",",
      "head" -> "false",
      "task" -> "classification")

    val configtest = Map("file" -> "data/ripleytest.csv",
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

    optModel.setMaxIterations(2).learn()

    val met = optModel.evaluate(configtest)

    met.print()
    met.generatePlots()

    println("Optimal Configuration: "+optConfig)
    val scale = if(logscale) "log" else "linear"

    val perf = met.kpi()
    val row = Seq(kernel, prototypes.toString, globalOptMethod,
      grid.toString, step.toString, scale,
      perf(0), perf(1), perf(2), optConfig.toString)

    val writer = CSVWriter.open(new File("data/resultsRipley.csv"), append = true)
    writer.writeRow(row)
    writer.close()
  }
}