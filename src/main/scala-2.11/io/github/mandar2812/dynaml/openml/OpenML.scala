package io.github.mandar2812.dynaml.openml

import java.io.{BufferedReader, FileReader}

import io.github.mandar2812.dynaml.dataformat.ArffFile
import io.github.mandar2812.dynaml.pipes.DataPipe
import org.apache.commons.io.FileUtils
import org.apache.log4j.Logger
import org.openml.apiconnector.io.OpenmlConnector
import org.openml.apiconnector.xml.{DataSetDescription, Task}

import scala.collection.JavaConversions
import scala.io.Source

/**
  * @author mandar date 07/09/16.
  *
  * The OpenML object is the one stop for interacting with
  * the OpenML api. It can be used to download and apply
  * DynaML flows on data sets and uplaoding the results
  * back to the OpenML server.
  */
object OpenML {

  private val logger = Logger.getLogger(this.getClass)

  val client: OpenmlConnector = new OpenmlConnector()

  private var _cacheLocation = "~/.openml/cache/"

  // Getter
  def cacheLocation = _cacheLocation

  // Setter
  def cacheLocation_= (value:String):Unit = _cacheLocation = value

  /**
    * Establish connection with openml server
    * and log in as per the user API Key provided
    * */
  def login(): Unit = {
    val standardIn = System.console()
    logger.info("Please enter your OpenML API Key Below: ")
    var apiKey = standardIn.readPassword()
    client.setApiKey(apiKey.mkString(""))
    apiKey = null
    logger.info("API key set")
  }

  def clearCache(): Unit = FileUtils.cleanDirectory(new java.io.File(cacheLocation))

  /**
    * Get the attibutes of a particular data set.
    *
    * @param id The numeric id of the data set in the OpenML data collection
    * */
  def dataset(id: Int): DataSetDescription = client.dataGet(id)

  /**
    * Download information on an OpenML task
    *
    * @param id The task id on the OpenML server.
    * */
  def task(id: Int):OpenMLTask = new OpenMLTask(client.taskGet(id))

  /**
    * Download an OpenML data set as a [[java.io.File]]
    * */
  def downloadDataSet(id: Int): java.io.File = dataset(id).getDataset(client.getApiKey)


  def loadDataIntoARFF(id: Int): ArffFile = {
    val arff = new ArffFile()
    arff.parse(new BufferedReader(new FileReader(downloadDataSet(id))))
    arff
  }


  /**
    * Download data set from OpenML and load the
    * text file into a Stream of lines.
    * */
  val openMLDataToStream = DataPipe((id: Int) => {
    Source.fromFile(downloadDataSet(id)).getLines().toStream
  })

}
