package com.huangshihe.aliexam01

import java.util.Date

import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}

/**
  * Created by huangshihe on 27/03/2017.
  *
  */
object Demo {
    val spark: SparkSession = SparkSession.builder().appName("Spark MLlib demo").master("local").getOrCreate()

    val sqlContext: SQLContext = spark.sqlContext

    val df: DataFrame = spark.read.csv("src/main/resources/data/tianchi_fresh_comp_train_user.csv")

    import spark.implicits._

    def main(args: Array[String]): Unit = {
        df.take(10).foreach(println)
    }

    /**
      * 行为类型：浏览-1，收藏-2，加购物车-3，购买-4
      * x: 第1-29天(2014-11-18 0~2014-12-17 23)中用户对该商品最后一次加购物车的(用户商品)数据
      * y: 第30天用户购买该商品的概率
      */
    def lastTime3(): Unit ={
        // 0. 确定对象为用户-商品对即：(user_id, item_id)
        // 1. 筛选出1-29天中最后一次用户加购物车操作
//        df.map(row=>row.getAs[Date](5).getTime).take(10).foreach(println)
//        val date29 = new Date("2014-12-17 23")
//        df.filter(row=> (row.getDate(6) date29))
        // 2. 数据去重
        // 3. 按小时对数据进行归类
        // 4. 统计每个小时中的记录数
        // 5. 筛选出第30天中用户购买的操作
        // 6. 数据去重（多次购物在这里只算一次购买记录）
        // 7. 计算概率，即在每次加购物车的小时中，成功购买的次数/当前小时中加购物车的次数

    }
}
