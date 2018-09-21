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
package io.github.mandar2812.dynaml.graphics

import org.jzy3d.analysis.{AnalysisLauncher, IAnalysis}
import org.jzy3d.colors.colormaps.{ColorMapRainbow, IColorMap}
import org.jzy3d.maths.Coord2d

/**
  * <h3>DynaML 3d Plotting API</h3>
  *
  * Contains the user API for rendering 3d surface plots
  * in a convenient fashion.
  *
  * The [[plot3d.draw()]] methods provide for generating
  * 3d surfaces.
  *
  * To render the image on the system GUI, call the
  * [[plot3d.show()]] method using the result returned by [[plot3d.draw()]]
  *
  * @author mandar2812 date: 2018/05/11
  * */
package object plot3d {

  /**
    * Generate a 3 dimensional surface from a function.
    * */
  def draw(
    function: (Double, Double) => Double,
    xAxisLimits: (Float, Float)   = (-3.0f, 3.0f),
    yAxisLimits: (Float, Float)   = (-3.0f, 3.0f),
    xAxisBins : Int               = 100,
    yAxisBins : Int               = 100,
    displayWireFrame: Boolean     = true,
    colorMap: IColorMap           = new ColorMapRainbow): Surface =
    new Surface(
      function, xAxisLimits, yAxisLimits,
      xAxisBins, yAxisBins, displayWireFrame,
      colorMap)

  /**
    * Generated a tessellated surface from a [[Stream]]
    * of x, y & z coordinates.
    * */
  def draw(
    points: Traversable[((Double, Double), Double)],
    colorMap: IColorMap): DelauneySurface =
    new DelauneySurface(
      points.map(p => ((p._1._1.toFloat, p._1._2.toFloat), p._2.toFloat)),
      colorMap)

  def draw(points: Traversable[((Double, Double), Double)]): DelauneySurface =
    new DelauneySurface(points.map(p => ((p._1._1.toFloat, p._1._2.toFloat), p._2.toFloat)))

  def draw(data: Traversable[(Double, Double)], numBins: Int): Histogram3D =
    new Histogram3D(data.map(d => new Coord2d(d._1.toFloat, d._2.toFloat)).toIterable, numBins)

  def draw(points: Traversable[(Double, Double, Double)], resolution: Int, drawPoints: Boolean): LinePlot3D =
    new LinePlot3D(points.map(p => (p._1.toFloat, p._2.toFloat, p._3.toFloat)), resolution, drawPoints)

  /**
    * Render a 3d surface on the system GUI.
    * */
  def show(chart: IAnalysis): Unit = AnalysisLauncher.open(chart)

}
