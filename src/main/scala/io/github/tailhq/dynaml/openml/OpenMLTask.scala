package io.github.tailhq.dynaml.openml

import java.io.{BufferedReader, FileReader}
import org.openml.apiconnector.xml.Task
import OpenML._
import io.github.tailhq.dynaml.dataformat.{ARFF, ArffFile}

/**
  * Created by mandar on 08/09/16.
  */
case class OpenMLTask(t: Task) {

  def inputs(): Array[Task#Input] = t.getInputs

  def getDataSplitsAsStream: Stream[String] = {
    val estimation_procedure_index = inputs().map(_.getName).indexOf("estimation_procedure")

    val splits = inputs()(estimation_procedure_index)
      .getEstimation_procedure.getDataSplits(t.getTask_id)

    val arff = new ArffFile()
    arff.parse(new BufferedReader(new FileReader(splits)))
    ARFF.asStream(arff)
  }

  def getDataAsStream: Stream[String] = {
    val data_index = inputs().map(_.getName).indexOf("source_data")
    val data_id = inputs()(data_index).getData_set.getData_set_id
    val data = dataset(data_id)
    data.getFormat match {
      case "ARFF" =>  ARFF.asStream(loadDataIntoARFF(data_id))
    }
  }

}
