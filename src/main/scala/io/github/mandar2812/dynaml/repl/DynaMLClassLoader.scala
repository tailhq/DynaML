package io.github.mandar2812.dynaml.repl

import scala.reflect.internal.util.ScalaClassLoader
import scala.tools.nsc.interpreter.JPrintWriter
import scala.tools.nsc.io.AbstractFile
import scala.tools.nsc.util

class DynaMLClassLoader(root:AbstractFile, parent:ClassLoader, out:JPrintWriter) extends
  util.AbstractFileClassLoader(root,parent) with ScalaClassLoader{
  override def loadClass(name: String): Class[_] = {
    //out.println(s"DynaMLClassLoader loads classOf ${name}")
    super.loadClass(name)
  }
}
