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

import org.jzy3d.analysis.AbstractAnalysis
import org.jzy3d.chart.factories.AWTChartComponentFactory
import org.jzy3d.colors.{Color, ColorMapper}
import org.jzy3d.colors.colormaps.ColorMapRainbow
import org.jzy3d.maths
import org.jzy3d.plot3d.builder.{Builder, Mapper}
import org.jzy3d.plot3d.builder.concrete.OrthonormalGrid
import org.jzy3d.plot3d.primitives.Shape
import org.jzy3d.plot3d.rendering.canvas.Quality

class DelauneySurface(function: (Double, Double) => Double) extends AbstractAnalysis {
  override def init(): Unit = { // Define a function to plot
    val mapper = new Mapper() {
      def f(x: Double, y: Double) = function(x, y)
    }
    // Define range and precision for the function to plot
    val range = new maths.Range(-3, 3)
    val steps = 80

    val grid = new OrthonormalGrid(range, steps, range, steps)

    // Create the object to represent the function over the given range.
    val surface: Shape = Builder.buildDelaunay(grid.apply(mapper))

    surface.setColorMapper(
      new ColorMapper(
        new ColorMapRainbow,
        surface.getBounds.getZmin,
        surface.getBounds.getZmax,
        new Color(1, 1, 1, .5f))
    )

    surface.setFaceDisplayed(true)

    // Create a chart
    chart = AWTChartComponentFactory.chart(Quality.Advanced, getCanvasType)
    chart.getScene.getGraph.add(surface)
  }
}
