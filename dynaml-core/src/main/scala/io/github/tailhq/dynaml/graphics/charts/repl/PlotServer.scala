package io.github.tailhq.dynaml.graphics.charts.repl

import unfiltered.request._
import unfiltered.response._

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Promise}

/**
 * User: austin
 * Date: 12/1/14
 *
 * An unfiltered web-app for displaying graphs
 */
class PlotServer extends UnfilteredWebApp[UnfilteredWebApp.Arguments]  {
  // this is fulfilled by the plot command, to allow a browser to wait for plot to reload
  var p = Promise[Unit]()

  private class WebApp extends unfiltered.filter.Plan {
    def intent = {
      // handle jsonp
      case req @ GET(Path(Seg("check" :: Nil)) & Params(params)) =>
        implicit val responder = req
        val str = """[]"""
        val response = params.get("callback") match {
          case Some(v) =>
            val callbackName = v.head
            s"$callbackName($str)"
          case _ => str
        }
        // block for plot command to fulfill promise, and release this result to trigger browser reload
        Await.result(p.future, Duration.Inf)
        JsonContent ~> ResponseString(response)
      case _ => Pass
    }
  }

  def parseArgs(args: Array[String]) = {
    val parsed = new UnfilteredWebApp.Arguments{}
    parsed.parse(args)
    parsed
  }

  def setup(parsed: UnfilteredWebApp.Arguments): unfiltered.filter.Plan = {
    new WebApp
  }

  def htmlRoot: String = "/"
}