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
import org.jzy3d.maths.Coord3d
import org.jzy3d.plot3d.builder.Builder
import org.jzy3d.plot3d.primitives.Shape
import org.jzy3d.plot3d.rendering.canvas.Quality

import scala.collection.JavaConverters._

/**
  * A surface described by a list of x,y & z coordinates.
  *
  * @param points A scala [[Stream]] of the data points.
  * @param colorMap A colour mapping which dictates how colours are varied
  *                 from lower to higher values of the plotted function.
  * @author tailhq date: 2018/05/11
  * */
class DelauneySurface(
  points: Traversable[((Float, Float), Float)],
  colorMap: IColorMap = new ColorMapRainbow) extends
  AbstractAnalysis {

  override def init(): Unit = {

    // Create the tessellated object.
    val surface: Shape = Builder.buildDelaunay(
      points.map(p => new Coord3d(p._1._1, p._1._2, p._2)).toList.asJava
    )

    surface.setColorMapper(
      new ColorMapper(
        colorMap,
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
