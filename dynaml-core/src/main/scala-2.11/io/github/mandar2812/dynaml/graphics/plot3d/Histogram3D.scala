/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
* */
package io.github.mandar2812.dynaml.graphics.plot3d

import io.github.mandar2812.dynaml.analysis.implicits._
import io.github.mandar2812.dynaml.utils
import org.jzy3d.analysis.AbstractAnalysis
import org.jzy3d.chart.factories.AWTChartComponentFactory
import org.jzy3d.colors.colormaps.{ColorMapRainbow, IColorMap}
import org.jzy3d.colors.{Color, ColorMapper}
import org.jzy3d.maths.{Coord2d, Coord3d}
import org.jzy3d.plot3d.primitives.HistogramBar
import org.jzy3d.plot3d.rendering.canvas.Quality
import scalaxy.streams.optimize


/**
  * Represents a 3D histogram.
  *
  * */
class Histogram3D(
  val data: Iterable[Coord2d],
  numBins: Int = 20,
  colorMap: IColorMap = new ColorMapRainbow)
  extends AbstractAnalysis{


  private val TOLERANCE = 1E-3

  override def init(): Unit = {
    /*
    * 1. Calculate minimum and maximum data ranges
    * 2. Construct bins
    * 3. Create Histogram object.
    * */

    val (x_min, x_max) = {
      val xs = data.map(_.getX)
      (xs.min, xs.max)
    }

    val (y_min, y_max) = {
      val ys = data.map(_.getY)
      (ys.min, ys.max)
    }


    val x_range = utils.range(x_min, x_max*(1 + TOLERANCE), numBins)
    val y_range = utils.range(y_min, y_max*(1 + TOLERANCE), numBins)

    val bins = utils.combine(
      Seq(
        x_range.sliding(2).toSeq.map(h => (h.head, h.last)),
        y_range.sliding(2).toSeq.map(h => (h.head, h.last)))
    ).map(
      s => Histogram3D.Range(xMin = s.head._1, yMin = s.last._1, xMax = s.head._2, yMax = s.last._2)
    )


    val hist: Map[Histogram3D.Range, Int] = {

      val dataAndBin = optimize { for(d <- data; b <- bins) yield (d, b)}

      optimize {
        dataAndBin.map(pattern => {
          val (data, bin) = pattern
          if(bin.contains(data.x, data.y)) (bin, 1) else (bin, 0)
        }).groupBy(_._1).map(gr => (gr._1, gr._2.map(_._2).sum))
          .filter(kv => kv._2 > 0)
      }
    }

    val col_mapper = new ColorMapper(
        colorMap,
        hist.values.min,
        hist.values.max,
        new Color(1, 1, 1, .5f))

    val histogram = new HistogramBar()

    hist.foreach(kv =>
      histogram.setData(
        new Coord3d(kv._1.avg._1.toFloat, kv._1.avg._2.toFloat, 0f),
        kv._2.toFloat, kv._1.size._1.toFloat/2f,
        col_mapper.getColor(kv._2.toFloat))
    )

    chart = AWTChartComponentFactory.chart(Quality.Advanced, getCanvasType)
    chart.getScene.getGraph.add(histogram)

  }
}

object Histogram3D {

  /**
    * Conceptual definition of a 3d histogram bin
    * */
  case class Range(xMin: Double, yMin: Double, xMax: Double, yMax: Double) {

    lazy val size: (Double, Double) = (xMax-xMin, yMax-yMin)

    lazy val avg: (Double, Double) = ((xMin+xMax)/2.0, (yMin+yMax)/2.0)

    /**
      * Returns true if argument lies in the range, à la Càdlàg
      * */
    def contains(x: Double, y: Double): Boolean = x >= xMin && x < xMax && y >= yMin && y < yMax

    def contains(x: Float, y: Float): Boolean = x >= xMin && x < xMax && y >= yMin && y < yMax


  }


}
