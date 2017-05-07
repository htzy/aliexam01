package com.huangshihe.aliexam01

import java.text.{DateFormat, SimpleDateFormat}
import java.util.Date

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}

/**
  * Created by huangshihe on 05/05/2017.
  */
object LogisticPredict {
    val spark: SparkSession = SparkSession.builder().appName("Tianmao logistic predict").master("local").getOrCreate()

    val sqlContext: SQLContext = spark.sqlContext

    // 日期格式："yyyy-MM-dd HH"，HH为0-23
    val df: DataFrame = spark.read.option("dateFormat", "yyyy-MM-dd HH").option("header", "true")
        .csv("src/main/resources/data/tianchi_fresh_comp_train_user.csv")

    def main(args: Array[String]): Unit = {
        //        df.take(10).foreach(println)
        getPredictData()
    }

    /**
      * 行为类型：浏览-1，收藏-2，加购物车-3，购买-4
      *
      */
    def getPredictData(): Unit = {
        // 将数据转换格式
        // ref: http://stackoverflow.com/questions/29383107/how-to-change-column-types-in-spark-sqls-dataframe
        df.printSchema()
        //        root
        //        |-- user_id: string (nullable = true)
        //        |-- item_id: string (nullable = true)
        //        |-- behavior_type: string (nullable = true)
        //        |-- user_geohash: string (nullable = true)
        //        |-- item_category: string (nullable = true)
        //        |-- time: string (nullable = true)

        // 以下方法中df的schema都不会发生变化
        // 方法一：使用org.apache.spark.sql.functions.udf + withColumn
        import org.apache.spark.sql.functions._
        val toInt = udf[Int, String](_.toInt)
        val changed = df.withColumn("user_id", toInt(df("user_id")))
        changed.printSchema()
        //        root
        //        |-- user_id: integer (nullable = true)
        //        |-- item_id: string (nullable = true)
        //        |-- behavior_type: string (nullable = true)
        //        |-- user_geohash: string (nullable = true)
        //        |-- item_category: string (nullable = true)
        //        |-- time: string (nullable = true)
        println("df:")
        df.printSchema()
        //        root
        //        |-- user_id: string (nullable = true)
        //        |-- item_id: string (nullable = true)
        //        |-- behavior_type: string (nullable = true)
        //        |-- user_geohash: string (nullable = true)
        //        |-- item_category: string (nullable = true)
        //        |-- time: string (nullable = true)
        // 方法二：用withColumn + cast
        df.withColumn("user_id", df("user_id").cast(IntegerType)).printSchema()
        println("df:")
        df.printSchema() // schema保持不变

        // 方法三：用select + cast + as
        df.select(df("user_id").cast(IntegerType).as("user_id"), df("item_id").cast(IntegerType).as("item_id")).printSchema()
        //        root
        //        |-- user_id: integer (nullable = true)
        //        |-- item_id: integer (nullable = true)
    }

    /**
      * "浏览"类型
      */
    val VIEW_TYPE = 1
    /**
      * "收藏"类型
      */
    val COLLECT_TYPE = 2
    /**
      * "加购物车"类型
      */
    val ADD_TO_CART_TYPE = 3
    /**
      * "购物"类型
      */
    val BUY_TYPE = 4

    import org.apache.spark.sql.functions._

    val day30: java.sql.Date = java.sql.Date.valueOf("2014-12-18")
    val day31: java.sql.Date = java.sql.Date.valueOf("2014-12-19")
    val simpleDateFormat: DateFormat = new SimpleDateFormat("yyyy-MM-dd HH")

    val toInt: UserDefinedFunction = udf[Int, String](_.toInt)
    val toHour: UserDefinedFunction = udf((t: String) => "%04d".format(t.toInt).take(2).toInt)
    val getBetweenHours: UserDefinedFunction =
        udf((big: String, small: Date) => (simpleDateFormat.parse(big).getTime - small.getTime) / (1000 * 60 * 60))

    // 这里的参数time，不能使用_代替，否则报错
    val getDiffHour: UserDefinedFunction =
        udf((time: String) => (day31.getTime - simpleDateFormat.parse(time).getTime) / (1000 * 60 * 60))

}
