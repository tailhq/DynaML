import org.kuleuven.esat.graphicalModels.GaussianLinearModel

/**
 * @author mandar2812
 */

val config = Map("file" -> "data/ripley.csv", "delim" -> ",",
  "head" -> "false",
  "task" -> "classification")

val configtest = Map("file" -> "data/ripleytest.csv",
  "delim" -> ",",
  "head" -> "false")

val model = GaussianLinearModel(config)

model.tuneRBFKernel(folds = 8)

val met = model.evaluate(configtest)

met.print()

met.generatePlots()
