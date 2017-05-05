package com.huangshihe.logisticregression

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, LabeledPoint, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession, functions}

/**
  * Created by huangshihe on 25/04/2017.
  */
object EmailLogisticRegression {

    val spark: SparkSession = SparkSession.builder().appName("email logistic regression").master("local").getOrCreate()

    val spam: RDD[String] = spark.sparkContext.textFile("src/main/resources/data/email_spam.txt")
    val normal: RDD[String] = spark.sparkContext.textFile("src/main/resources/data/email_normal.txt")

    def useML(): Unit = {
        // 开启隐式转换，否则rdd不能直接调用toDF转为DataFrame
        import spark.implicits._

        // 首先使用分解器Tokenizer把句子划分为单个词语
        val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")

        val spamWordsData = tokenizer.transform(spam.toDF("sentence"))
        val hamWordsData = tokenizer.transform(normal.toDF("sentence"))

        // 使用HashingTF将单词转换为特征向量，最后使用IDF重新调整特征向量。这种转换通常可以提高使用文本特征的性能。
        val hashingTF = new HashingTF().setInputCol("words").setOutputCol("features").setNumFeatures(10000)

        val spamFeatures = hashingTF.transform(spamWordsData)
        val hamFeatures = hashingTF.transform(hamWordsData)

        spamFeatures.take(10).foreach(println)
        hamFeatures.take(10).foreach(println)

        // 创建LabeledPoint数据集分别存放阳性（垃圾邮件）和阴性（正常邮件）的例子
        val positiveExample = spamFeatures.map(row => LabeledPoint(1, row.getAs[org.apache.spark.ml.linalg.Vector]("features")))
        val negativeExample = hamFeatures.map(row => LabeledPoint(0, row.getAs[org.apache.spark.ml.linalg.Vector]("features")))
        val trainingData = positiveExample.union(negativeExample)

        trainingData.cache()

        trainingData.take(10).foreach(println)
        trainingData.printSchema()

        val lor = new LogisticRegression()
        val model = lor.fit(trainingData)

        val posTest = hashingTF.transform(tokenizer.transform(Seq("omg get cheap stuff by sending money to...").toDF("sentence")))
        val negTest = hashingTF.transform(tokenizer.transform(Seq("Hi Dad, I started studying Spark the other ...").toDF("sentence")))

        model.transform(posTest).show()
        model.transform(negTest).show()
        // print schema
        //        model.transform(posTest).printSchema()
        //        root
        //        |-- sentence: string (nullable = true)
        //        |-- words: array (nullable = true)
        //        |    |-- element: string (containsNull = true)
        //        |-- features: vector (nullable = true)
        //        |-- rawPrediction: vector (nullable = true)
        //        |-- probability: vector (nullable = true)
        //        |-- prediction: double (nullable = true)
    }

    def useMLWithPipeline(): Unit = {
        // 开启隐式转换，否则rdd不能直接调用toDF转为DataFrame
        import spark.implicits._

        val data = spam.toDF("sentence").withColumn("label", functions.lit(1.0)) union
            normal.toDF("sentence").withColumn("label", functions.lit(0.0))
        data.show()

        // 首先使用分解器Tokenizer把句子划分为单个词语
        val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
        // 使用HashingTF将单词转换为特征向量，最后使用IDF重新调整特征向量。这种转换通常可以提高使用文本特征的性能。
        val hashingTF = new HashingTF().setInputCol(tokenizer.getOutputCol).setOutputCol("features").setNumFeatures(10000)
        // 使用逻辑回归
        val lor = new LogisticRegression()
        // 新建pipeline，分为三个阶段：分词、转换特征向量和逻辑回归
        val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lor))
        // 训练模型
        val model = pipeline.fit(data)

        // spark.createDataFrame可以通过Seq创建df，但是Seq中的数据格式是tuples，不能直接是String，Int等，例子如下：
        // 以下是正确的
        //        val testOk = spark.createDataFrame(Seq(
        //            (1L, "apache spark")
        //        )).toDF("id", "sentence")
        //        val testError = spark.createDataFrame(Seq("apache spark")).toDF("sentence")

        // 测试数据转为dataframe
        val test = Seq("omg get cheap stuff by sending money to ...", "Hi Dad, I started studying Spark the other ...").toDF("sentence")

        //        model.transform(test).show()
        model.transform(test)
            .select("sentence", "probability", "prediction")
            .collect()
            .foreach { case Row(sentence: String, prob: org.apache.spark.ml.linalg.Vector, prediction: Double) =>
                println(s"($sentence) --> prob=$prob, prediction=$prediction")
            }
    }

    def useMLWithParam(): Unit = {
        // 开启隐式转换，否则rdd不能直接调用toDF转为DataFrame
        import spark.implicits._

        val data = spam.toDF("sentence").withColumn("label", functions.lit(1.0)) union
            normal.toDF("sentence").withColumn("label", functions.lit(0.0))
        data.show()

        // 首先使用分解器Tokenizer把句子划分为单个词语
        val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
        // 使用HashingTF将单词转换为特征向量，最后使用IDF重新调整特征向量。这种转换通常可以提高使用文本特征的性能。
        val hashingTF = new HashingTF().setInputCol(tokenizer.getOutputCol).setOutputCol("features").setNumFeatures(10000)
        // 使用逻辑回归
        val lor = new LogisticRegression()

        val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lor))

        // 通过交叉验证对一批参数进行网格搜索，来找到最佳的模型，而不是使用pipeline.fit(data)只对训练集进行一次拟合
        val paramMaps = new ParamGridBuilder()
            .addGrid(hashingTF.numFeatures, Array(10000, 20000))
            .addGrid(lor.maxIter, Array(100, 200))
            .build()
        val eval = new BinaryClassificationEvaluator()
        val cv = new CrossValidator().setEstimatorParamMaps(paramMaps).setEstimator(pipeline).setEvaluator(eval)
        val bestModel = cv.fit(data)

        // 测试数据转为dataframe
        val test = Seq("omg get cheap stuff by sending money to ...", "Hi Dad, I started studying Spark the other ...").toDF("sentence")

        //        model.transform(test).show()
        bestModel.transform(test)
            .select("sentence", "probability", "prediction")
            .collect()
            .foreach { case Row(sentence: String, prob: org.apache.spark.ml.linalg.Vector, prediction: Double) =>
                println(s"($sentence) --> prob=$prob, prediction=$prediction")
            }
    }

    @Deprecated
    def useMLLib(): Unit = {
        // 创建一个HashingTF实例把邮件文本映射为包含10000个特征的向量
        val tf = new org.apache.spark.mllib.feature.HashingTF(numFeatures = 10000)
        // 各邮件都被切分为单词，每个单词被映射为一个特征
        val spamFeatures = spam.map(email => tf.transform(email.split(" ")))
        val normalFeatures = normal.map(email => tf.transform(email.split(" ")))
        // 创建LabeledPoint数据集分别存放阳性（垃圾邮件）和阴性（正常邮件）的例子
        val positiveExamples = spamFeatures.map(features => org.apache.spark.mllib.regression.LabeledPoint(1, features))
        val negativeExamples = normalFeatures.map(features => org.apache.spark.mllib.regression.LabeledPoint(0, features))
        val trainingData = positiveExamples.union(negativeExamples)
        trainingData.cache()
        // 因为逻辑回归是迭代算法，所以缓存训练数据rdd
        // 使用SGD算法运行逻辑回归
        val model = new org.apache.spark.mllib.classification.LogisticRegressionWithSGD().run(trainingData)

        // 以阳性（垃圾邮件）和阴性（正常邮件）的例子分别进行测试
        val posTest = tf.transform("omg get cheap stuff by sending money to...".split(" "))
        val negTest = tf.transform("Hi Dad, I started studying Spark the other ...".split(" "))
        println("Prediction for positive test example:" + model.predict(posTest))
        println("Prediction for negative test example:" + model.predict(negTest))

    }

    def main(args: Array[String]): Unit = {
        // Spark2.0 推荐使用ml而不是mllib
        //        useMLLib()
        //        useML()
        //        useMLWithPipeline()
        useMLWithParam()
        //        demoParam()
        spark.stop()
    }

    /**
      * https://spark.apache.org/docs/2.0.2/ml-tuning.html
      */
    def demoParam(): Unit = {
        val training = spark.createDataFrame(Seq(
            (0L, "a b c d e spark", 1.0),
            (1L, "b d", 0.0),
            (2L, "spark f g h", 1.0),
            (3L, "hadoop mapreduce", 0.0),
            (4L, "b spark who", 1.0),
            (5L, "g d a y", 0.0),
            (6L, "spark fly", 1.0),
            (7L, "was mapreduce", 0.0),
            (8L, "e spark program", 1.0),
            (9L, "a e c l", 0.0),
            (10L, "spark compile", 1.0),
            (11L, "hadoop software", 0.0)
        )).toDF("id", "text", "label")

        // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
        val tokenizer = new Tokenizer()
            .setInputCol("text")
            .setOutputCol("words")
        val hashingTF = new HashingTF()
            .setInputCol(tokenizer.getOutputCol)
            .setOutputCol("features")
        val lr = new LogisticRegression()
            .setMaxIter(10)
        val pipeline = new Pipeline()
            .setStages(Array(tokenizer, hashingTF, lr))

        // We use a ParamGridBuilder to construct a grid of parameters to search over.
        // With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
        // this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
        val paramGrid = new ParamGridBuilder()
            .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
            .addGrid(lr.regParam, Array(0.1, 0.01))
            .build()

        // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
        // This will allow us to jointly choose parameters for all Pipeline stages.
        // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
        // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
        // is areaUnderROC.
        val cv = new CrossValidator()
            .setEstimator(pipeline)
            .setEvaluator(new BinaryClassificationEvaluator)
            .setEstimatorParamMaps(paramGrid)
            .setNumFolds(2) // Use 3+ in practice

        // Run cross-validation, and choose the best set of parameters.
        val cvModel = cv.fit(training)

        // Prepare test documents, which are unlabeled (id, text) tuples.
        val test = spark.createDataFrame(Seq(
            (4L, "spark i j k"),
            (5L, "l m n"),
            (6L, "mapreduce spark"),
            (7L, "apache hadoop")
        )).toDF("id", "text")

        // Make predictions on test documents. cvModel uses the best model found (lrModel).
        cvModel.transform(test)
            .select("id", "text", "probability", "prediction")
            .collect()
            .foreach { case Row(id: Long, text: String, prob: org.apache.spark.ml.linalg.Vector, prediction: Double) =>
                println(s"($id, $text) --> prob=$prob, prediction=$prediction")
            }
    }
}
