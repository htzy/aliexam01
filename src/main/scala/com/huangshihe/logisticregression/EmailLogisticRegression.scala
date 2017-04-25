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

        val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
        import spark.implicits._

        tokenizer.transform(spam.toDF("sentence")).take(10).foreach(println)

    }
}
