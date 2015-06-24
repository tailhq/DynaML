import org.kuleuven.esat.graphicalModels.GaussianLinearModel
import org.kuleuven.esat.optimization.GridSearch

/**
 * @author mandar2812
 */
object TestRipley {
  def apply(kernel: String): Unit = {
    val config = Map("file" -> "data/ripley.csv", "delim" -> ",",
      "head" -> "false",
      "task" -> "classification")

    val configtest = Map("file" -> "data/ripleytest.csv",
      "delim" -> ",",
      "head" -> "false")

    val model = GaussianLinearModel(config)

    val gs = new GridSearch(model)

    val (optModel, optConfig) = kernel match {
      case "RBF" => gs.optimize(Map("bandwidth" -> 1.0, "RegParam" -> 0.5),
        Map("kernel" -> "RBF"))

      case "Polynomial" => gs.optimize(Map("degree" -> 1.0, "offset" -> 1.0, "RegParam" -> 0.5),
        Map("kernel" -> "Polynomial"))

      case "Exponential" => gs.optimize(Map("beta" -> 1.0, "RegParam" -> 0.5),
        Map("kernel" -> "Exponential"))
    }

    optModel.setMaxIterations(5).learn()

    val met = optModel.evaluate(configtest)

    met.print()

    met.generatePlots()
  }
}