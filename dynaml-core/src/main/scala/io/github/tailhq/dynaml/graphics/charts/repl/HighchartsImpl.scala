package io.github.tailhq.dynaml.graphics.charts.repl

import java.io.{ PrintWriter, File }

import io.github.tailhq.dynaml.graphics.charts.highcharts.Highchart
import scala.concurrent.Promise
import scala.util.Random
import org.apache.commons.io.FileUtils
/**
 * User: austin
 * Date: 12/3/14
 */
trait WebPlotHighcharts extends WebPlot[Highchart] {

  /**
   * Iterates through the plots and builds the necessary javascript and html around them.
   * returns the files contents as a string
   */
  def buildHtmlFile(): String = {
    val sb = new StringBuilder()
    sb.append(jsHeader)
    sb.append(reloadJs)
    sb.append("</head>")
    sb.append("<body>")
    plots.map(highchartsContainer).foreach(sb.append)
    sb.append("</body>")
    sb.append("</html>")

    sb.toString()
  }

  def plotAll(): Unit = {

    val fileContents = buildHtmlFile()

    val temp = File.createTempFile("highcharts", ".html")
    val pw = new PrintWriter(temp)
    pw.print(fileContents)
    pw.flush()
    pw.close()

    plotServer.foreach { ps =>
      ps.p.success(())
      ps.p = Promise[Unit]()
    }

    val (serverRootFile, port, serverMode) = getWispServerInfo()

    lazy val link =
      if (serverMode) {
        FileUtils.deleteQuietly(serverRootFile)
        FileUtils.moveFile(temp, serverRootFile)
        s"http://${java.net.InetAddress.getLocalHost.getCanonicalHostName}:${port}"
      } else s"file://$temp"

    openFirstWindow(link)

    println(s"Output written to $link (CMD + Click link in Mac OSX).")
  }

  override def plot(t: Highchart): Highchart = {
    super.plot(t)
    plots = t +: plots
    undoStack.push(plots)
    plotAll()
    t
  }

  def reloadJs =
    """
      |<script type="text/javascript">$.ajax({url: '/check', dataType: 'jsonp', complete: function(){location.reload()}})</script>
    """.stripMargin

  val wispJsImports: String =
    """
      |<script type="text/javascript" src="http://code.jquery.com/jquery-1.8.2.min.js"></script>
      |<script type="text/javascript" src="http://code.highcharts.com/4.0.4/highcharts.js"></script>
      |<script type="text/javascript" src="http://code.highcharts.com/4.0.4/modules/exporting.js"></script>
      |<script type="text/javascript" src="http://code.highcharts.com/4.0.4/highcharts-more.js"></script>
    """.stripMargin

  val jsHeader =
    """
      |<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
      |<html>
      |  <head>
      |    <title>
      |      Highchart
      |    </title>
      |    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    """.stripMargin +
      wispJsImports

  def highchartsContainer(hc: Highchart): String = {
    val hash = hc.hashCode()
    val containerId = Random.nextInt(1e10.toInt) + (if (hash < 0) -1 else 1) * hash // salt the hash to allow duplicates
    highchartsContainer(hc.toJson, containerId)
  }

  def highchartsContainer(json: String, index: Int): String =
    containerDivs(index) + "\n" +
      """
      |    <script type="text/javascript">
      |        $(function() {
      |            $('#container%s').highcharts(
    """.stripMargin.format(index.toString) +
      """
      |                %s
      |            );
      |        });
      |    </script>
      |
    """.stripMargin.format(json)

  def containerDivs(index: Int) =
    s"""
      |    <div id="container%s" style="min-width: 400px; height: 400px; margin: 0 auto"></div>
    """.stripMargin.format(index.toString)
}
