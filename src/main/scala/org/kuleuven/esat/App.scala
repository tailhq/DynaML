package org.kuleuven.esat

import java.io.File
import com.github.tototoshi.csv._
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

    implicit object MyFormat extends DefaultCSVFormat {
      override val delimiter = delim
      override val quoting = QUOTE_NONNUMERIC
    }

    val reader = CSVReader.open(new File(args.apply(0)))

    val model = GaussianLinearModel(reader)

    model.setLearningRate(0.001)
      .setMaxIterations(args.apply(2).toInt)
      .learn

    println(model.parameters())
  }

}
