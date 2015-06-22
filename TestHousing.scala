import org.kuleuven.esat.graphicalModels.GaussianLinearModel

/**
 * @author mandar2812 on 22/6/15.
 */

object TestHousing {
  def apply(): Unit = {
    val config = Map("file" -> "data/bostonhousing.csv", "delim" -> ",",
      "head" -> "false",
      "task" -> "regression")

    val configtest = Map("file" -> "data/bostonhousingtest.csv",
      "delim" -> ",",
      "head" -> "false")

    val model = GaussianLinearModel(config)

    model.tuneRBFKernel(folds = 8)

    val met = model.evaluate(configtest)

    met.print()

    met.generatePlots()
  }
}

