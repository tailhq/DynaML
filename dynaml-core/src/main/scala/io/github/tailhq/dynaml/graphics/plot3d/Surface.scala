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
package io.github.tailhq.dynaml.graphics.plot3d

import org.jzy3d.analysis.AbstractAnalysis
import org.jzy3d.chart.factories.AWTChartComponentFactory
import org.jzy3d.colors.{Color, ColorMapper}
import org.jzy3d.colors.colormaps.{ColorMapRainbow, IColorMap}
import org.jzy3d.maths
import org.jzy3d.plot3d.builder.{Builder, Mapper}
import org.jzy3d.plot3d.builder.concrete.OrthonormalGrid
import org.jzy3d.plot3d.primitives.Shape
import org.jzy3d.plot3d.rendering.canvas.Quality

/**
  * Represents a surface described by a function of 2 variables.
  *
  * @param function Surface to be plotted.
  * @param xAxisLimits Lower and upper limits of the x-Axis
  * @param yAxisLimits Lower and upper limits of the y-Axis
  * @param xAxisBins Number of divisions of the x-Axis.
  * @param yAxisBins Number of divisions of the y-Axis
  * @param displayWireFrame Set to true, if wireframe is to be displayed
  *                         showing the grid cells.
  * @param colorMap A colour mapping which dictates how colours are varied
  *                 from lower to higher values of the plotted function.
  * @author tailhq date: 2018/05/11
  * */
class Surface(
  function: (Double, Double) => Double,
  xAxisLimits: (Float, Float)   = (-3.0f, 3.0f),
  yAxisLimits: (Float, Float)   = (-3.0f, 3.0f),
  xAxisBins : Int               = 100,
  yAxisBins : Int               = 100,
  displayWireFrame: Boolean     = true,
  colorMap: IColorMap           = new ColorMapRainbow) extends
  AbstractAnalysis {

  override def init(): Unit = {
    // Define a function to plot
    val mapper = new Mapper() {
      def f(x: Double, y: Double) = function(x, y)
    }

    val (x_min, x_max) = xAxisLimits
    val (y_min, y_max) = yAxisLimits
    // Define range and precision for the function to plot
    val (range_x, range_y) = (
      new maths.Range(x_min, x_max),
      new maths.Range(y_min, y_max))


    // Create the object to represent the function over the given range.
    val surface: Shape = Builder.buildOrthonormal(
      new OrthonormalGrid(range_x, xAxisBins, range_y, yAxisBins),
      mapper
    )

    surface.setColorMapper(
      new ColorMapper(
        colorMap,
        surface.getBounds.getZmin,
        surface.getBounds.getZmax,
        new Color(1, 1, 1, .5f))
    )

    surface.setFaceDisplayed(true)
    surface.setWireframeDisplayed(displayWireFrame)
    // Create a chart
    chart = AWTChartComponentFactory.chart(Quality.Advanced, getCanvasType)
    chart.getScene.getGraph.add(surface)
  }
}
