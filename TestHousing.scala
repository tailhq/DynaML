import com.tinkerpop.blueprints.Graph
import com.tinkerpop.frames.FramedGraph
import org.kuleuven.esat.graphUtils.CausalEdge
import org.kuleuven.esat.models.GaussianLinearModel
import org.kuleuven.esat.optimization.GridSearch

/**
 * @author mandar2812 on 22/6/15.
 */

object TestHousing {
  def apply(kernel: String): Unit = {
    val config = Map("file" -> "data/bostonhousing.csv", "delim" -> ",",
      "head" -> "false",
      "task" -> "regression")

    val configtest = Map("file" -> "data/bostonhousingtest.csv",
      "delim" -> ",",
      "head" -> "false")

    val model = GaussianLinearModel(config)

    val gs = new GridSearch[FramedGraph[Graph], Iterable[CausalEdge], model.type](model)

    val (optModel, optConfig) = kernel match {
      case "RBF" => gs.optimize(Map("bandwidth" -> 1.0, "RegParam" -> 0.5),
        Map("kernel" -> "RBF"))

      case "Polynomial" => gs.optimize(Map("degree" -> 1.0, "offset" -> 1.0, "RegParam" -> 0.5),
        Map("kernel" -> "Polynomial"))

      case "Exponential" => gs.optimize(Map("beta" -> 1.0, "RegParam" -> 0.5),
        Map("kernel" -> "Exponential"))

      case "Laplacian" => gs.optimize(Map("beta" -> 1.0, "RegParam" -> 0.5),
        Map("kernel" -> "Laplacian"))
    }

    optModel.setMaxIterations(5).learn()

    val met = optModel.evaluate(configtest)

    met.print()
  }
}

