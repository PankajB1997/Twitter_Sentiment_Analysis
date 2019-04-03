import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassifier, NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.feature.{HashingTF, IDF, StopWordsRemover, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.ml.feature.{IndexToString, StandardScaler, StringIndexer}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}

//class Dropper(override val uid: String) extends Transformer {
//
//  def this() = this(Identifiable.randomUID("dropper"))
//
//  override def transform(dataset: Dataset[_]): DataFrame = {
//    dataset.drop("your-column-name-here")
//  }
//
//  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
//
//  override def transformSchema(schema: StructType): StructType = {
//    //here you should right your result schema i.e. the schema without the dropped column
//  }
//
//}

object SentimentAnalysis {

  def setupLogging() = {
    import org.apache.log4j.{Level, Logger}
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
  }

  def main(args: Array[String]): Unit = {
    // Ronald

    val spark = SparkSession.builder.
      master("local")
      .appName("example")
      .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    val df = spark.read.format("csv").option("header", "true").
      load("/Users/ronaldlim/Downloads/Twitter_Sentiment_Analysis-master/preprocessor/src/main/resources/sample_tweets_with_sentiments.csv")
//
//    val df = spark.read.format("csv").option("header", "true").
//      load("/Users/ronaldlim/Downloads/Twitter_Sentiment_Analysis-master/preprocessor/src/main/resources/sample.csv")

    val tokenizer = new Tokenizer().setInputCol("tweet").setOutputCol("words")
    val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol("words").setOutputCol("rawFeatures")
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(2)
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)

    val newCol = when(df("anger") =!= 0, 1).when(df("anger") === 0, 0)
    val newCol2 = when(df("positive") =!= 0, 1).when(df("positive") === 0, 0)
    val newCol3 = when(df("anticipation") =!= 0, 1).when(df("anticipation") === 0, 0)
    val newCol4 = when(df("disgust") =!= 0, 1).when(df("disgust") === 0, 0)
    val newCol5 = when(df("fear") =!= 0, 1).when(df("fear") === 0, 0)
    val newCol6 = when(df("joy") =!= 0, 1).when(df("joy") === 0, 0)
    val newCol7 = when(df("sadness") =!= 0, 1).when(df("sadness") === 0, 0)
    val newCol8 = when(df("surprise") =!= 0, 1).when(df("surprise") === 0, 0)
    val newCol9 = when(df("trust") =!= 0, 1).when(df("trust") === 0, 0)
    val newCol10 = when(df("negative") =!= 0, 1).when(df("negative") === 0, 0)

    val df2 = df.withColumn("anger", newCol)
      .withColumn("positive", newCol2)
      .withColumn("anticipation", newCol3)
      .withColumn("disgust", newCol4)
      .withColumn("fear", newCol5)
      .withColumn("joy", newCol6)
      .withColumn("sadness", newCol7)
      .withColumn("surprise", newCol8)
      .withColumn("trust", newCol9)
      .withColumn("negative", newCol10)
      .toDF()

//    df2.show()

    val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3))

    val lr1 = new LogisticRegression()
      .setMaxIter(10)
      .setTol(1E-6)
      .setFitIntercept(true)
      .setLabelCol("anger")
      .setFeaturesCol("scaledFeatures")
      .setProbabilityCol("anger_prob")
      .setRawPredictionCol("anger_raw_pred")
      .setPredictionCol("anger_pred")

    val lr2 = new LogisticRegression()
      .setMaxIter(10)
      .setTol(1E-6)
      .setFitIntercept(true)
      .setLabelCol("anticipation")
      .setFeaturesCol("scaledFeatures")
      .setProbabilityCol("anticipation_prob")
      .setRawPredictionCol("anticipation_raw_pred")
      .setPredictionCol("anticipation_pred")

    val layers = Array[Int](1000, 500, 250, 1)


    val mlp1 = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
      .setLabelCol("anger")
      .setFeaturesCol("scaledFeatures")
      .setProbabilityCol("anger_prob")
      .setRawPredictionCol("anger_raw_pred")
      .setPredictionCol("anger_pred")

    val mlp2 = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
      .setLabelCol("anticipation")
      .setFeaturesCol("scaledFeatures")
      .setProbabilityCol("anticipation_prob")
      .setRawPredictionCol("anticipation_raw_pred")
      .setPredictionCol("anticipation_pred")

    // 10 different models
    val rf1 = new RandomForestClassifier()
      .setLabelCol("anger")
      .setFeaturesCol("scaledFeatures")
      .setProbabilityCol("anger_prob")
      .setRawPredictionCol("anger_raw_pred")
      .setPredictionCol("anger_pred")
      .setNumTrees(100)
    val rf2 = new RandomForestClassifier()
      .setLabelCol("anticipation")
      .setFeaturesCol("scaledFeatures")
      .setProbabilityCol("anticipation_prob")
      .setRawPredictionCol("anticipation_raw_pred")
      .setPredictionCol("anticipation_pred")
      .setNumTrees(100)
    val rf3 = new RandomForestClassifier()
      .setLabelCol("disgust")
      .setFeaturesCol("scaledFeatures")
      .setProbabilityCol("disgust_prob")
      .setRawPredictionCol("disgust_raw_pred")
      .setPredictionCol("disgust_pred")
      .setNumTrees(100)
    val rf4 = new RandomForestClassifier()
      .setLabelCol("fear")
      .setFeaturesCol("scaledFeatures")
      .setProbabilityCol("fear_prob")
      .setRawPredictionCol("fear_raw_pred")
      .setPredictionCol("fear_pred")
      .setNumTrees(100)
    val rf5 = new RandomForestClassifier()
      .setLabelCol("joy")
      .setFeaturesCol("scaledFeatures")
      .setProbabilityCol("joy_prob")
      .setRawPredictionCol("joy_raw_pred")
      .setPredictionCol("joy_pred")
      .setNumTrees(100)
    val rf6 = new RandomForestClassifier()
      .setLabelCol("sadness")
      .setFeaturesCol("scaledFeatures")
      .setProbabilityCol("sadness_prob")
      .setRawPredictionCol("sadness_raw_pred")
      .setPredictionCol("sadness_pred")
      .setNumTrees(100)
    val rf7 = new RandomForestClassifier()
      .setLabelCol("surprise")
      .setFeaturesCol("scaledFeatures")
      .setProbabilityCol("surprise_prob")
      .setRawPredictionCol("surprise_raw_pred")
      .setPredictionCol("surprise_pred")
      .setNumTrees(100)
    val rf8 = new RandomForestClassifier()
      .setLabelCol("trust")
      .setFeaturesCol("scaledFeatures")
      .setProbabilityCol("trust_prob")
      .setRawPredictionCol("trust_raw_pred")
      .setPredictionCol("trust_pred")
      .setNumTrees(100)
    val rf9 = new RandomForestClassifier()
      .setLabelCol("negative")
      .setFeaturesCol("scaledFeatures")
      .setProbabilityCol("negative_prob")
      .setRawPredictionCol("negative_raw_pred")
      .setPredictionCol("negative_pred")
      .setNumTrees(100)
    val rf10 = new RandomForestClassifier()
      .setLabelCol("positive")
      .setFeaturesCol("scaledFeatures")
      .setProbabilityCol("positive_prob")
      .setRawPredictionCol("positive_raw_pred")
      .setPredictionCol("positive_pred")
      .setNumTrees(100)

//    val labelIndexer1 = new StringIndexer()
//      .setInputCol("prediction")
//      .setOutputCol("anticipation_label")
//
//    val labelIndexer2 = new StringIndexer()
//      .setInputCol("prediction")
//      .setOutputCol("anger_label")

//    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, idf, scaler,
//      rf1, rf9))

//    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, idf, scaler,
//      rf1, rf2, rf3, rf4, rf5, rf6, rf7, rf8, rf9, rf10))

    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, idf, scaler,
      mlp1, mlp2))

//    val paramMap2 = ParamMap(rf1.predictionCol -> "prediction_anger")

    val model = pipeline.fit(trainingData)
    val predictions = model.transform(testData)
    predictions.select(
      "anger",
      "anger_pred",
      "anticipation",
      "anticipation_pred"
//      "disgust",
//      "disgust_pred",
//      "fear",
//      "fear_pred",
//      "joy",
//      "joy_pred",
//      "sadness",
//      "sadness_pred",
//      "surprise",
//      "surprise_pred",
//      "trust",
//      "trust_pred",
//      "negative",
//      "negative_pred",
//      "positive",
//      "positive_pred"
    ).show()

    val evaluator1 = new MulticlassClassificationEvaluator()
          .setLabelCol("anger")
          .setPredictionCol("anger_pred")
          .setMetricName("accuracy")
    val anger_accuracy = evaluator1.evaluate(predictions)
    println("Anger Test Error = " + (1.0 - anger_accuracy))
    println()
    
    val evaluator2 = new MulticlassClassificationEvaluator()
      .setLabelCol("anticipation")
      .setPredictionCol("anticipation_pred")
      .setMetricName("accuracy")
    val anticipation_accuracy = evaluator2.evaluate(predictions)
    println("Anticipation Test Error = " + (1.0 - anticipation_accuracy))
    println()
//
//    val evaluator3 = new MulticlassClassificationEvaluator()
//      .setLabelCol("disgust")
//      .setPredictionCol("disgust_pred")
//      .setMetricName("accuracy")
//    val disgust_accuracy = evaluator3.evaluate(predictions)
//    println("Disgust Test Error = " + (1.0 - disgust_accuracy))
//    println()
//
//    val evaluator4 = new MulticlassClassificationEvaluator()
//      .setLabelCol("fear")
//      .setPredictionCol("fear_pred")
//      .setMetricName("accuracy")
//    val fear_accuracy = evaluator4.evaluate(predictions)
//    println("Fear Test Error = " + (1.0 - fear_accuracy))
//    println()
//
//    val evaluator5 = new MulticlassClassificationEvaluator()
//      .setLabelCol("joy")
//      .setPredictionCol("joy_pred")
//      .setMetricName("accuracy")
//    val joy_accuracy = evaluator5.evaluate(predictions)
//    println("Joy Test Error = " + (1.0 - joy_accuracy))
//    println()
//
//    val evaluator6 = new MulticlassClassificationEvaluator()
//      .setLabelCol("sadness")
//      .setPredictionCol("sadness_pred")
//      .setMetricName("accuracy")
//    val sadness_accuracy = evaluator6.evaluate(predictions)
//    println("Sadness Test Error = " + (1.0 - sadness_accuracy))
//    println()
//
//    val evaluator7 = new MulticlassClassificationEvaluator()
//      .setLabelCol("surprise")
//      .setPredictionCol("surprise_pred")
//      .setMetricName("accuracy")
//    val surprise_accuracy = evaluator7.evaluate(predictions)
//    println("Surprise Test Error = " + (1.0 - surprise_accuracy))
//    println()
//
//    val evaluator8 = new MulticlassClassificationEvaluator()
//      .setLabelCol("trust")
//      .setPredictionCol("trust_pred")
//      .setMetricName("accuracy")
//    val trust_accuracy = evaluator8.evaluate(predictions)
//    println("Trust Test Error = " + (1.0 - trust_accuracy))
//    println()
//
//    val evaluator9 = new MulticlassClassificationEvaluator()
//      .setLabelCol("negative")
//      .setPredictionCol("negative_pred")
//      .setMetricName("accuracy")
//    val negative_accuracy = evaluator9.evaluate(predictions)
//    println("Negative Test Error = " + (1.0 - negative_accuracy))
//    println()
//
//    val evaluator10 = new MulticlassClassificationEvaluator()
//      .setLabelCol("positive")
//      .setPredictionCol("positive_pred")
//      .setMetricName("accuracy")
//    val positive_accuracy = evaluator10.evaluate(predictions)
//    println("Positive Test Error = " + (1.0 - positive_accuracy))

    }
}