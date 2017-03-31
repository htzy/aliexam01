package com.huangshihe.aliexam01

import java.text.{DateFormat, SimpleDateFormat}
import java.util.Date

import org.apache.spark.sql._

/**
  * Created by huangshihe on 27/03/2017.
  *
  */
object Demo {
    val spark: SparkSession = SparkSession.builder().appName("Spark MLlib demo").master("local").getOrCreate()

    import spark.implicits._

    val sqlContext: SQLContext = spark.sqlContext

    // 日期格式："yyyy-MM-dd HH"，HH为0-23
    val df: DataFrame = spark.read.option("dateFormat", "yyyy-MM-dd HH").option("header", "true")
        .csv("src/main/resources/data/tianchi_fresh_comp_train_user.csv")

    def main(args: Array[String]): Unit = {
        df.take(10).foreach(println)
    }

    case class UserItemTime(user_id: String, item_id: String, time: Date)

    val simpleDateFormat: DateFormat = new SimpleDateFormat("yyyy-MM-dd HH")


    /**
      * 行为类型：浏览-1，收藏-2，加购物车-3，购买-4
      * x: 第1-29天(2014-11-18 0~2014-12-17 23)中用户对该商品最后一次加购物车的(用户商品)数据
      * y: 第30天用户购买该商品的概率
      */
    def lastTime3(): Unit = {
        // 0. 确定对象为用户-商品对即：(user_id, item_id)

        // 1. 筛选出1-29天中最后一次用户加购物车操作(3)
        val dayDate30: java.sql.Date = java.sql.Date.valueOf("2014-12-18")
        var lastAdd = df.filter(row => simpleDateFormat.parse(row.getString(5)).before(dayDate30))
            .filter(row => 3.equals(Integer.valueOf(row.getString(2))))

        // 2. 数据去重
        // 去重掉无用的列，这里也去除behavior_type列，因为可以确保这里的均为3
        lastAdd = lastAdd.drop("user_geohash", "item_category", "behavior_type")
        //        println(lastAdd.count)//640950
        // 去除原有的time，新增一列day，即所有的数据均为29天的集合中的数据，最后去重
        // lastAdd = lastAdd.drop("time").withColumn("day", functions.lit(29)).distinct()
        //        println(lastAdd.count)//527522
        // 需要保留的数据为对象+小时，并将对象去重(当出现两个一样的key时，抛弃第一个：x，只要第二个：y)
        val lastAddData = lastAdd.map(row => (
            (row.getAs[String]("user_id"), row.getAs[String]("item_id")),
            getBetweenHours(dayDate30, simpleDateFormat.parse(row.getAs[String]("time"))).toInt))
            .rdd.reduceByKey((x, y) => y)        // count:527522
        // 3. 按小时对数据进行归类
        // 将key和value互换，'小时'作为key，'用户-商品对'作为value
        val groupByHours = lastAddData.map(row => (row._2, row._1)).groupByKey()
        // 4. 统计每个小时中的记录数(hours, counts)
        val countsByHours = groupByHours.map(row => (row._1, row._2.size))
        // 5. 筛选出第30天中用户购买(4)的操作
        val finalBuy = df.filter(row => simpleDateFormat.parse(row.getString(5)).after(dayDate30))
            .filter(row => 4.equals(Integer.valueOf(row.getString(2))))
        // 6. 数据去重（多次购物在这里只算一次购买记录）
        // 去除无用列
        val finalBuyData = finalBuy.drop("user_geohash", "item_category", "behavior_type", "time").distinct()
            .map(row => (row.getAs[String]("user_id"), row.getAs[String]("item_id")))
        //        println(finalBuyData.count()) // 5976
        // 7. 计算概率，即在每次加购物车的小时中，成功购买的次数/当前小时中加购物车的次数
        // 生成格式：(hours，该小时内加入购物车并与最后一天成功购买的比例)
        // 分布式不能传参
//        val result = groupByHours.map(row => (row._1, getSuccessPer(finalBuyData, row._2))).take(10).foreach(println)
        val result = groupByHours.map(row => (row._1, finalBuyData.intersect(row._2.toStream.toDS()).count())).take(10).foreach(println)
    }

    def getBetweenHours(big: Date, small: Date): Long = {
        (big.getTime - small.getTime) / (1000 * 60 * 60)
    }

    /**
      * 计算成功的概率
      *
      * @param finalBuy 最后一天成功购买的总记录数
      * @param hourAdd  最后时刻（hours）增加到购物车中的记录
      * @return
      */
    def getSuccessPer(finalBuy: Dataset[(String, String)], hourAdd: Iterable[(String, String)]): Double = {
        if (hourAdd.isEmpty) {
            return 0.0
        }
        val stream = hourAdd.toStream
        val ds = stream.toDS
        val success = finalBuy.intersect(ds)
        (success.count() + 0.0) / hourAdd.size
    }

}
