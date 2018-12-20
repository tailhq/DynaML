package io.github.mandar2812.dynaml.openml

import org.openml.apiconnector.xml.Run

/**
  * Created by mandar on 08/09/16.
  */
case class OpenMLRun(r: Run) {
  def upload(): Unit = {
    //Upload the run
  }
}

object OpenMLRun {

  def apply(task_id: Int, error_message: String, flow_id: Int,
            setup_string: String, parameter_settings: Array[Run.Parameter_setting],
            tags: Array[String]) =
    new OpenMLRun(
      new Run(
        task_id, error_message, flow_id,
        setup_string, parameter_settings, tags))
}