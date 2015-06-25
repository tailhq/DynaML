import java.io.File
import com.github.tototoshi.csv.CSVWriter
import com.tinkerpop.blueprints.Graph
import com.tinkerpop.frames.FramedGraph
import org.kuleuven.esat.graphUtils.CausalEdge
import org.kuleuven.esat.graphicalModels.GaussianLinearModel
import org.kuleuven.esat.optimization.{CoupledSimulatedAnnealing, GridSearch}

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

    val nProt = if(prototypes > 0) prototypes else math.sqrt(model.npoints.toDouble).toInt

    val gs = globalOptMethod match {
      case "gs" => new GridSearch[FramedGraph[Graph],
        Iterable[CausalEdge], model.type](model).setGridSize(grid)
        .setStepSize(step).setLogScale(logscale)

      case "csa" => new CoupledSimulatedAnnealing[FramedGraph[Graph],
        Iterable[CausalEdge], model.type](model).setGridSize(grid)
        .setStepSize(step).setLogScale(logscale).setMaxIterations(5)
    }

    val (optModel, optConfig) = kernel match {
      case "RBF" => gs.optimize(Map("bandwidth" -> 1.0, "RegParam" -> 0.5),
        Map("kernel" -> "RBF", "subset" -> prototypes.toString))

      case "Polynomial" => gs.optimize(Map("degree" -> 1.0, "offset" -> 1.0, "RegParam" -> 0.5),
        Map("kernel" -> "Polynomial", "subset" -> prototypes.toString))

      case "Exponential" => gs.optimize(Map("beta" -> 1.0, "RegParam" -> 0.5),
        Map("kernel" -> "Exponential", "subset" -> prototypes.toString))

      case "Laplacian" => gs.optimize(Map("beta" -> 1.0, "RegParam" -> 0.5),
        Map("kernel" -> "Laplacian", "subset" -> prototypes.toString))
    }

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