package io.github.tailhq.dynaml.graphics.charts.repl

import java.io.File
import scala.util.{Failure, Try}
import unfiltered.util.Port
import unfiltered.jetty.Server

import scala.collection.mutable

/**
 * User: austin
 * Date: 11/14/14
 *
 * Defines a plotting api for interoperability with PlotServer.
 * Stores the plots in a stack
 */
trait PlotLike[T] {
  def plot(t: T): T
}

trait Plottable[T] extends PlotLike[T] {
  var plots = List[T]()

  def plotAll()

  // Heavy handed approach to undo / redo - maintain entire state in stack
  // It is up to the plot-library author that their calls to plot()
  // interoperate with undo/redo
  protected val undoStack = new mutable.Stack[List[T]]()
  protected val redoStack = new mutable.Stack[List[T]]()

  def undo() = {
    if(undoStack.nonEmpty) {
      redoStack.push(undoStack.pop())
      plots = if(undoStack.nonEmpty) undoStack.head else List[T]()
      plotAll()
    }
  }

  def redo = {
    if(redoStack.nonEmpty) {
      undoStack.push(redoStack.pop())
      plots = undoStack.head
      plotAll()
    }
  }

  def delete() = {
    if(plots.nonEmpty) {
      undoStack.push(plots)
      plots = plots.tail
      plotAll()
    }
  }

  def deleteAll() = {
    undoStack.push(plots)
    plots = List[T]()
    plotAll()
  }
}

/**
 * Maintains the server statex
 * @tparam T : a Plotting type : Highchart or Vega
 */
trait WebPlot[T] extends Plottable[T] {

  private var serverRootFileName = s"index-${System.currentTimeMillis()}.html"
  private var port = Port.any
  private var serverMode = false
  private var firstOpenWindow = false

  private var serverRootFile = new File(serverRootFileName)

  var http: Option[Server] = None
  var plotServer: Option[PlotServer] = None

  startWispServer()

  /**
   *
   * @return
   */
  def getWispServerInfo(): (File, Int, Boolean) = {
    (serverRootFile, port, serverMode)
  }

  def setWispServerFile(filename: String): Unit = {
    stopWispServer
    this.serverRootFileName = filename
    this.serverRootFile = new File(serverRootFileName)
    startWispServer()
  }

  def setWispServerFile(file: File): Unit = {
    setWispServerFile(file.getAbsolutePath())
  }

  def setWispPort(port: Int): Unit = {
    stopWispServer
    this.port = port
    startWispServer()
  }

  def disableOpenWindow(): Unit = {
    this.firstOpenWindow = true
  }

  def openWindow(link: String) = {
   import sys.process._
    Try{
      java.awt.Desktop.getDesktop.browse(new java.net.URI(link))
      link
    }
    .orElse(Try(s"open $link"!!))
    .orElse(Try(s"xdg-open $link"!!))
  }

  /**
   * If this is the first plot command being called, try to open the browser
   * @param link
   */
  def openFirstWindow(link: String) = {
    if(!firstOpenWindow) {
      openWindow(link) match {
        case Failure(msg) =>
          println(s"Error while opening window (cause: $msg)")
          println(s"You can browse the following URL: $link")
        case _ =>
      }
      firstOpenWindow = true
    }
  }

  /**
   * Launches the server which hosts the plots. InetAddress.getLocalHost requires a properly configured /etc/hosts
   * on linux machines.
   * Assigns a random port
   * @param message
   */
  def startWispServer(message: String = s"http://${java.net.InetAddress.getLocalHost.getCanonicalHostName}:${port}/${serverRootFileName}") {
    if (!serverMode) {
      serverMode = true
      val ps = new PlotServer
      val args = ps.parseArgs(Array("--altRoot", serverRootFile.getAbsolutePath, "--port", port.toString))
      val server = ps.get(args)
      server.start
      println("Server started: " + message)
      http = Some(server)
      plotServer = Some(ps)
    }
  }

  /**
   * Deletes the resulting index-*.html and stops the server
   * Currently the index-*.html file persists in the $cwd if stopServer is not called.
   */
  def stopWispServer {
    if (serverMode) {
      serverRootFile.delete()
      // satisfy the promise, to avoid exception on close
      // TODO handle failure in the PlotServer
      plotServer.map(_.p.success(()))
      http.map(_.stop)
      http.map(_.destroy)
      serverMode = false
      plotServer = None
    }
  }

  def plot(t: T): T = {
    startWispServer()
    t
  }
}

/**
  * Chart augmenting functions:
  *  - hold (from Matlab syntax) - until unhold is called, graphs will be plotted on the same plot, ie the series will be merged into one Highchart
  *  - unhold: stop holding. Next graph will be on its own plot
  *  - xlabel: assigns a label to the x-axis of the most recent plot
  *  - ylabel: assign a label to the y-axis of the most recent plot
  *
  * */
trait Hold[T] extends PlotLike[T] {
  var isHeld: Boolean = false
  def hold(): Unit = {
    isHeld = true
  }
  def unhold(): Unit = {
    isHeld = false
  }
}

trait Labels[T] extends PlotLike[T] {
  def xAxis(label: String): T
  def yAxis(label: String): T
  def title(label: String): T
  def legend(labels: Iterable[String]): T
}
