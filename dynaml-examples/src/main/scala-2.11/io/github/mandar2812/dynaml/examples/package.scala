package io.github.mandar2812.dynaml

/**
  * Created by mandar on 20/06/2017.
  */
package object examples {

  val dynaml_install_dir = System.getenv("DYNAML_HOME")

  val root_dir = if (dynaml_install_dir != null) dynaml_install_dir else "."

  def dataDir = root_dir+"/data"
}
