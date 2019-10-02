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
    <groupId>com.github.transcendent-ai-labs</groupId>
    <artifactId>DynaML</artifactId>
    <version>v1.4</version>
</dependency>
```

## SBT

For sbt projects edit your `build.sbt` (see [JitPack](https://jitpack.io/#transcendent-ai-labs/DynaML) for more details)

```scala
    resolvers += "jitpack" at "https://jitpack.io"
    libraryDependencies += "com.github.transcendent-ai-labs" % "DynaML" % version
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
