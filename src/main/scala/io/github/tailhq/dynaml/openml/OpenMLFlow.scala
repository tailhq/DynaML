package io.github.tailhq.dynaml.openml

import io.github.tailhq.dynaml.pipes.DataPipe
import org.openml.apiconnector.xml.Flow

/**
  * @author tailhq date 08/09/16.
  *
  * Wrapper around the [[Flow]] class
  * of the OpenML Java API. An OpenML flow
  * is also a DynaML pipe which takes as input
  * a task and outputs
  */
case class OpenMLFlow(f: Flow) extends DataPipe[OpenMLTask, Option[OpenMLRun]]{

  /**
    * Wrapper around the addComponent method of [[Flow]]
    *
    * @param identifier Name of the component
    * @param implementation The [[Flow]] instance
    * @param updateName Should the name of the parent [[Flow]] be updated,
    *                   defaults to true
    * */
  def addComponent(identifier: String,
                   implementation: Flow,
                   updateName: Boolean = true): Unit = {
    f.addComponent(identifier, implementation, updateName)
  }

  /**
    * Wrapper around addParameter method of [[Flow]]
    *
    * @param name Name of the said parameter
    * @param data_type Data type of parameter
    * @param description Optional string describing parameter
    *                    and its role in the flow
    * */
  def addParameter(name: String, data_type: String,
                   default_value: String,
                   description: String = ""): Unit = {
    f.addParameter(name, data_type, default_value, description)
  }


  /**
    * Take as input an OpenML task [[OpenMLTask]]
    * and if possible return an OpenML run [[OpenMLRun]]
    * @todo Complete implementation, currently only a stub exists
    * */
  override def run(data: OpenMLTask): Option[OpenMLRun] = None
}


object OpenMLFlow {

  /**
    * Create a new OpenML flow
    * */
  def apply(name: String, external_version: String, description: String,
            language: String, dependencies: String): OpenMLFlow =
    new OpenMLFlow(new Flow(
      name, external_version, description,
      language, dependencies)
    )

  /**
    * Create a new OpenML flow, with more options
    * */
  def apply(name: String, external_version: String, description: String,
            creator: Array[String], contributor: Array[String], licence: String,
            language: String, full_description: String,
            installation_notes: String, dependencies: String,
            tag: Array[String]): OpenMLFlow =
    new OpenMLFlow(
      new Flow(name, external_version, description,
        creator, contributor, licence,
        language, full_description,
        installation_notes, dependencies, tag)
    )
}