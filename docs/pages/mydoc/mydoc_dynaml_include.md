---
title: Import
tags: [getting_started, troubleshooting]
keywords:
summary: "DynaML can also be imported into your JVM project as dependency, the artifacts are uploaded to JitPack from where you can pull them using sbt, maven, gradle and leinengen"
sidebar: mydoc_sidebar
permalink: mydoc_dynaml_include.html
folder: mydoc
---

## Maven

To include DynaML in your maven JVM project edit your ```pom.xml``` file as follows

```xml
<repositories>
   <repository>
	   <id>jitpack.io</id>
	   <url>https://jitpack.io</url>
	 </repository>
</repositories>
```

```xml
<dependency>
    <groupId>com.github.mandar2812</groupId>
    <artifactId>DynaML</artifactId>
    <version>v1.4-beta.3</version>
</dependency>
```

## SBT

For sbt projects edit your `build.sbt` (see [JitPack](https://jitpack.io/#mandar2812/DynaML) for more details)

```scala
    resolvers += "jitpack" at "https://jitpack.io"
    libraryDependencies += "com.github.User" % "Repo" % "Tag"
```

## Gradle

In your gradle project, add the following to the root `build.gradle` as follows

```groovy
allprojects {
  repositories {
    ...
    maven { url "https://jitpack.io" }
  }
}
```

and then add the dependency like

```groovy
dependencies {
	compile 'com.github.User:Repo:Tag'
}
```

## Leinengen

In `project.clj`

```clojure
:repositories [["jitpack" "https://jitpack.io"]]
```

```clojure
:dependencies [[com.github.User/Repo "Tag"]]
```
