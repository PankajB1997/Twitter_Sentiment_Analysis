name := "TwitterSentimentAnalysis"

version := "0.1"

scalaVersion := "2.11.0"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.4.0"
libraryDependencies += "org.apache.spark" %% "spark-hive" % "2.4.0"
libraryDependencies += "com.microsoft.sqlserver" % "mssql-jdbc" % "6.1.0.jre7"