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

import org.jzy3d.analysis.{AbstractAnalysis, AnalysisLauncher, IAnalysis}
import org.jzy3d.colors.colormaps.{ColorMapRainbow, IColorMap}

/**
  * <h3>DynaML 3d Plotting API</h3>
  *
  * Contains the user API for rendering 3d surface plots
  * in a convenient fashion.
  * */
package object plot3d {

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

  def show(chart: IAnalysis): Unit = AnalysisLauncher.open(chart)

}
