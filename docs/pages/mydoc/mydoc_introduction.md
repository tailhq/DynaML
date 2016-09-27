---
title: Introduction
sidebar: mydoc_sidebar
permalink: mydoc_introduction.html
folder: mydoc
toc: false
---

## Motivation

DynaML was born out of the need to have a performant, extensible and easy to use Machine Learning research environment. Scala was a natural choice for these requirements due to its sprawling data science ecosystem (i.e. [Apache Spark](http://spark.apache.org/)), its functional object-oriented duality and its interoperability with the Java Virtual Machine.

## Organization

  <div class="row">
    <div class="col-lg-12">
        <h2 class="page-header">Modules</h2>
    </div>
    <div class="col-lg-12">

        <ul id="myTab" class="nav nav-tabs nav-justified">
            <li class="active"><a href="#service-one" data-toggle="tab"><i class="fa fa-tree"></i> Core API </a>
            </li>
            <li class=""><a href="#service-two" data-toggle="tab"><i class="fa fa-car"></i> Pipes API </a>
            </li>
            <li class=""><a href="#service-three" data-toggle="tab"><i class="fa fa-support"></i> REPL </a>
            </li>
            <li class=""><a href="#service-four" data-toggle="tab"><i class="fa fa-database"></i> REPL Examples </a>
            </li>
        </ul>

<div id="myTabContent" class="tab-content">

<div class="tab-pane fade active in" id="service-one">
<h4>Core</h4>

<p>
The core api consists of :

<ol>
  <li>Model implementations</li>
  <li>Optimization solvers</li>
  <li>Probability distributions/random variables</li>
  <li>Kernel functions for Non parametric models</li>
</ol>

</p>

<p markdown="1">To dive further into the different model classes supported in the core api start [here]({{site.baseurl}}/core_model_hierarchy.html)
</p>

</div>

<div class="tab-pane fade" id="service-two">
<h4>Pipes & Workflows</h4>

<p markdown="1">
The [pipes]({{site.baseurl}}/p1_pipes.html) module aims to separate model pre-processing tasks such as cleaning data files, replacing missing or corrupt records, applying transformations on data etc:
</p>

<p>
<ol>
  <li>Ability to create arbitrary workflows from scala functions and join them</li>
  <li>Feature transformations such as wavelet transform, gaussian scaling, auto-encoders etc</li>
</ol>
</p>

</div>

<div class="tab-pane fade" id="service-three">
<h4>DynaML Shell</h4>
<p markdown="1">The _read evaluate print loop_ (REPL) gives the user the ability to experiment with the data pre-processing and model building process in a mix and match fashion.</p>
<p markdown="1">The DynaML shell is based on the [Ammonite](http://www.lihaoyi.com/Ammonite/) project which is an augmented Scala REPL, all the features of the Ammonite REPL are a part of the DynaML REPL. </p>
</div>

<div class="tab-pane fade" id="service-four">
<h4>REPL Examples</h4>
<p markdown="1">The module ```dynaml-examples``` contains programs which build regression and classification models on various data sets. These examples serve as case studies as well as instructional material to show the capabilities of DynaML in a hands on manner. Click [here]({{site.baseurl}}/p2_examples.html) to get started with the examples.</p>

</div>
</div>
</div>
</div>


## Libraries Used

DynaML leverages a number of open source projects and builds on their useful features.

* [Breeze](https://github.com/scalanlp/breeze) for linear algebra operations with vectors, matrices etc.
* [Gremlin](https://github.com/tinkerpop/gremlin) for building graphs in Neural network based models.
* [Spire](https://github.com/non/spire) for creating algebraic entities like Fields, Groups etc.
* [Ammonite](http://www.lihaoyi.com/Ammonite/) for the shell environment.
* DynaML uses the newly minted [Wisp](https://github.com/quantifind/wisp) plotting library to generate aesthetic charts of common model validation metrics. In version 1.4 there is also integration of [plotly](https://plot.ly) which can now be imported and used directly in the shell environment.

![plots]({{site.baseurl}}/images/plot-screen.png)

{% include links.html %}
