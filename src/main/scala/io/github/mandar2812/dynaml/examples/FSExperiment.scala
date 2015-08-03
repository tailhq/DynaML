package io.github.mandar2812.dynaml.examples

/**
 * Created by mandar on 3/8/15.
 */
object FSExperiment {
  def apply(nCores: Int = 4, trials: Int, data: String = "ForestCover", root: String = "data/"): Unit = {
    List("gs", "csa").foreach((globalOpt) => {
      List(50, 100, 200, 300, 500).foreach((prototypes) => {
        List("RBF", "Polynomial", "Laplacian").foreach((kern) => {
          List(2, 3).foreach((gridSize) => {
            (1 to trials).foreach((trial) => {
              data match {
                case "ForestCover" =>
                  TestForestCover(nCores,
                    prototypes, kern,
                    globalOpt, grid = gridSize,
                    frac = 1.0, dataRoot = root)
                case "Adult" =>
                  TestAdult(nCores,
                    prototypes, kern,
                    globalOpt, grid = gridSize,
                    frac = 1.0)
                case "MagicGamma" =>
                  TestMagicGamma(nCores,
                    prototypes, kern,
                    globalOpt, grid = gridSize,
                    dataRoot = root)
              }

            })
          })
        })
      })
    })
  }
}
