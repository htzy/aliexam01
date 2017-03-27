package com.huangshihe.aliexam01

import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by huangshihe on 27/03/2017.
  */
object Demo {
    val spark = SparkSession.builder().appName("Spark MLlib demo").master("local").getOrCreate()

    val sqlContext = spark.sqlContext

    val df: DataFrame = spark.read.csv("src/main/resources/data/tianchi_fresh_comp_train_user.csv")

    def main(args: Array[String]): Unit = {
        df.take(10).foreach(println)
    }
}
