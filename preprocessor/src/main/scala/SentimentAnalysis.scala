import org.apache.spark.SparkContext

object SentimentAnalysis {

  def setupLogging() = {
    import org.apache.log4j.{Level, Logger}
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
  }

  def main(args: Array[String]): Unit = {
    // val context = new SparkContext("local[*]", "SentimentAnalysis")
    val context = SparkContext.getOrCreate()
    val lines = context.textFile("dbfs:/FileStore/tables/tweets_large_32M.csv")
    val stopwords = context.textFile("dbfs:/FileStore/text/stopwords.txt")
    val outputfile = "dbfs:/FileStore/output/preprocessed_tweets.csv"

    // Removing the @words
    val parsedHash = lines.map(x => x.replaceAll(Utility.REG_HANDLERS, ""))

    // Removing the links
    val parsedLinks = parsedHash.map(x => x.replaceAll(Utility.REG_LINKS, ""))

    // Removing punctuations
    val parsedPunctuations = parsedLinks.map(x => x.replaceAll(Utility.REG_PUNCTUATIONS, ""))

    // Lowercase
    val parsedLowercase = parsedPunctuations.map(x => x.toLowerCase)

    // Expanding contractions
    val parsedContractions = parsedLowercase.map(x => x.split(" ")
      .map(y => if (Utility.contractions.contains(y)) Utility.contractions(y) else y).mkString(" "))

    // Removing stopwords
    val broadcastStopwords = context.broadcast(stopwords.collect.toSet)
    val parsedStopwords = parsedContractions.map(x => x.split(" ")
      .map(y => if (!broadcastStopwords.value.contains(y)) y else "").mkString(" "))

    // Removing extra whitespaces
    val parsedWhitespaces = parsedStopwords.map(x => x.replaceAll(Utility.REG_WHITESPACES, " "))

    // Trimming the string
    val parsedTrim = parsedWhitespaces.map(x => x.replaceAll(Utility.REG_TRIM, ""))

    parsedTrim.saveAsTextFile(outputfile)

    // var count = 0
    // for (result <- parsedTrim.collect() if count < 5) {
    //   println(result)
    //   count = count + 1
    // }

    // Ronald

    val spark = SparkSession.builder.
      master("local")
      .appName("example")
      .getOrCreate()

    // Insert csv filename here
    val df = spark.read.format("csv").option("header", "true").load("test.csv")
    //    val df = spark.read.csv("file:///Users/ronaldlim/Downloads/trainingandtestdata/training.1600000.processed.noemoticon.csv").toDF("label", "col2", "col3", "col4", "col5", "text")
    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))

    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol("words").setOutputCol("rawFeatures")
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0)
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(df)
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("scaledFeatures")
      .setNumTrees(100)
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)


    //      val wordsData = tokenizer.transform(df)
    //
    //      val featurizedData = hashingTF.transform(wordsData)
    //
    //      val idfModel = idf.fit(featurizedData)
    //
    //      val rescaledData = idfModel.transform(featurizedData)
    //      rescaledData.select("label", "features").show()
    //
    //      val scalerModel = scaler.fit(rescaledData)
    //
    //      // Normalize each feature to have unit standard deviation.
    //      val scaledData = scalerModel.transform(rescaledData)
    //      scaledData.show()


    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, idf, labelIndexer, scaler, rf, labelConverter))

    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)
    //    predictions.select("predictedLabel", "label", "scaledFeatures").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    // Save model
//    model.write.overwrite().save("/tmp/spark-random-forest-model")
    // Load model
//    val sameModel = PipelineModel.load("/tmp/spark-logistic-regression-model")
  }
}