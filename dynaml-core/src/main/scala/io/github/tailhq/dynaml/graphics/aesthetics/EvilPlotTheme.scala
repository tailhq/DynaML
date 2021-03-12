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
package io.github.tailhq.dynaml.graphics.aesthetics

import com.cibo.evilplot.colors.{Color, HSL, HTMLNamedColors, RGB}
import com.cibo.evilplot.plot.aesthetics.DefaultTheme.{DefaultColors, DefaultElements, DefaultFonts}
import com.cibo.evilplot.plot.aesthetics.{Colors, Elements, Fonts, Theme}

object EvilPlotTheme {
  val ClassicColors: Colors = DefaultColors.copy(
    background = HTMLNamedColors.white,
    frame = RGB(30, 30, 30),
    bar = HSL(0, 0, 35),
    fill = HTMLNamedColors.white,
    path = HSL(0, 0, 0),
    point = HSL(0, 0, 35),
    gridLine = HTMLNamedColors.white,
    trendLine = HSL(0, 0, 35),
    title = HTMLNamedColors.black,
    label = HTMLNamedColors.black,
    annotation = HTMLNamedColors.black,
    legendLabel = HTMLNamedColors.black,
    tickLabel = HTMLNamedColors.black,
    stream = Color.stream
  )

  val ClassicElements: Elements = DefaultElements.copy(
    pointSize = 2.5,
    gridLineSize = 1,
    categoricalXAxisLabelOrientation = 90
  )

  val ClassicFonts: Fonts = DefaultFonts.copy(
    tickLabelSize = 10,
    legendLabelSize = 10
  )

  implicit val classicTheme: Theme = Theme(
    colors = ClassicColors,
    elements = ClassicElements,
    fonts = ClassicFonts
  )
}
