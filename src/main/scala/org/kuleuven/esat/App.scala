package org.kuleuven.esat

import org.kuleuven.esat.graphicalModels.GaussianLinearModel

/**
 * Hello world!
 *
 */
object App extends App {

  override def main(args: Array[String]): Unit = {
    //Read csv file
   var delim: Char = ','
    if(args.apply(1).compare("tab") == 0) delim = '\t'
    val model = GaussianLinearModel(
      utils.getCSVReader(args.apply(0), delim),
      true,  "regression")

    model.setLearningRate(0.001)
      .setMaxIterations(args.apply(2).toInt)
      .learn

    println(model.parameters())
  }

}
