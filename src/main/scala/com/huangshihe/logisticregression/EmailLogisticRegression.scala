package com.huangshihe.logisticregression

import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
  * Created by huangshihe on 25/04/2017.
  */
object EmailLogisticRegression {

    val spark: SparkSession = SparkSession.builder().appName("email logistic regression").master("local").getOrCreate()

    def main(args: Array[String]): Unit = {

        val spam: RDD[String] = spark.sparkContext.textFile("src/main/resources/data/email_spam.txt")
        val ham: RDD[String] = spark.sparkContext.textFile("src/main/resources/data/email_normal.txt")

        // 首先使用分解器Tokenizer把句子划分为单个词语
        val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
        // 对于每个句子（词袋），使用HashingTF将句子转换为特征向量，最后使用IDF重新调整特征向量。这种转换通常可以提高使用文本特征的性能。

        // 开启隐式转换，否则rdd不能直接调用toDF转为DataFrame
        import spark.implicits._

        tokenizer.transform(spam.toDF("sentence")).take(10).foreach(println)

    }
}
