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

    //model.tuneRBFKernel(folds = 8)
    val gs = new GridSearch(model)

    val (optModel, optConfig) = kernel match {
      case "RBF" => gs.optimize(Map("RegParam" -> 0.5, "bandwidth" -> 1.0),
        Map("kernel" -> "RBF"))

      case "Polynomial" => gs.optimize(Map("RegParam" -> 0.5, "degree" -> 1.0, "offset" -> 1.0),
        Map("kernel" -> "Polynomial"))
    }

    val met = optModel.evaluate(configtest)

    met.print()

    met.generatePlots()
  }
}