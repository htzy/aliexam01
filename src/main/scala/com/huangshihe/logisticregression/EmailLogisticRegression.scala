package com.huangshihe.logisticregression

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, LabeledPoint, Tokenizer}
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
        useMLWithPipeline()
        spark.stop()
    }
}
