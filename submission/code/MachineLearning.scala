import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, StandardScaler, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object MachineLearning {

  def main(args: Array[String]): Unit = {
    // Initialize Spark Session
    val spark = SparkSession.builder.
      master("local[*]")
      .appName("SentimentModel")
      .getOrCreate()
    import spark.implicits._
    spark.sparkContext.setLogLevel("ERROR")

    // Set root filepath
    val root = "dbfs:/FileStore/ml/"

    // Loading dataset
    var df = spark.read.format("csv").option("header", "true").
      load(root.concat(args(0)))

    // Initialising different stages in the pipeline
    val tokenizer = new Tokenizer().setInputCol("tweet").setOutputCol("words")
    val hashingTF = new HashingTF().setNumFeatures(10000).setInputCol("words").setOutputCol("rawFeatures")
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(2)
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)

    // Preprocessing for sentiment labels
    for (label <- Utility.LABELS) {
      df = df.withColumn(label, when(df(label) =!= 0, 1).when(df(label) === 0, 0))
    }

    // Split dataset into training and testing
    val Array(trainingData, testData) = df.randomSplit(Array(0.75, 0.25))

    // Set Pipeline stages
    val stages: Array[PipelineStage] = Array(tokenizer, hashingTF, idf, scaler)

    // Initialize some parameters
    val layers = Array[Int](10000, 512, 256, 128, 2)
    var predicted_results: Array[String] = Array()

    // Train a separate model for each sentiment, each label is a sentiment
    for (label <- Utility.LABELS) {
      // Initialize Deep Neural Network model
      val clf = new MultilayerPerceptronClassifier()
        .setLayers(layers)
        .setBlockSize(256)
        .setSeed(1234L)
        .setLabelCol(label)
        .setFeaturesCol("scaledFeatures")
        .setProbabilityCol(label.concat("_prob"))
        .setRawPredictionCol(label.concat("_raw_pred"))
        .setPredictionCol(label.concat("_pred"))

      // Initialize hyperparameter values for grid search
      val paramMaps = new ParamGridBuilder()
        .addGrid(hashingTF.numFeatures, Array(100, 1000, 10000))
        .addGrid(clf.regParam, Array(0.3, 0.1, 0.01))
        .addGrid(clf.maxIter, Array(10, 20, 50))
        .addGrid(clf.elasticNetParam, Array(0.0, 0.1, 0.2))
        .build()

      // Add Deep Neural Network model to pipeline
      val updatedStages = stages :+ clf
      val pipeline = new Pipeline().setStages(updatedStages)

      // Initialize evaluator for estimating model acuracy
      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol(label)
        .setPredictionCol(label.concat("_pred"))
        .setMetricName("accuracy")

      // Initialize K-fold Cross Validation, using 3 folds
      val cv = new CrossValidator()
        .setEstimator(pipeline)
        .setEstimatorParamMaps(paramMaps)
        .setEvaluator(evaluator)
        .setNumFolds(3)
        .setParallelism(2)

      // Train the model using grid search and 3-fold cross validation
      val model = cv.fit(trainingData)

      // Record predictions on test dataset
      val predictions = model.transform(testData)

      // Calculate accuracy on predicted results
      val accuracy = evaluator.evaluate(predictions) * 100
      predicted_results = predicted_results :+ label.concat(" accuracy " + accuracy)

      // Save trained model to HDFS
      model.save(root.concat("model_".concat(label)))
    }
    // Print accuracy figures for each sentiment's model
    predicted_results.foreach(println)
  }
}
