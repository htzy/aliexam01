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

    //    case class UserItemTime(user_id: String, item_id: String, time: Date)

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
        // 需要保留的数据为对象+小时，并将对象去重(当出现两个一样的key时，抛弃第一个：x，只要第二个：y) // count:527522
        val lastAddData = lastAdd.map(row => (
            (row.getAs[String]("user_id"), row.getAs[String]("item_id")),
            getBetweenHours(dayDate30, simpleDateFormat.parse(row.getAs[String]("time"))).toInt))
            .rdd.reduceByKey((_, y) => y)
        // 3. 按小时对数据进行归类
        // 将key和value互换，'小时'作为key，'用户-商品对'作为value
        val groupByHours = lastAddData.map(row => (row._2, row._1)).groupByKey()
        // 4. 统计每个小时中的记录数(hour, counts)
        val countsByHours = groupByHours.map(row => (row._1, row._2.size))
        // 5. 筛选出第30天中用户购买(4)的操作
        val finalBuy = df.filter(row => simpleDateFormat.parse(row.getString(5)).after(dayDate30))
            .filter(row => 4.equals(Integer.valueOf(row.getString(2))))
        // 6. 数据去重（多次购物在这里只算一次购买记录）
        // 去除无用列
        val finalBuyData = finalBuy.drop("user_geohash", "item_category", "behavior_type", "time")
            .distinct().withColumn("count", functions.lit(1)).map(row =>
            ((row.getAs[String]("user_id"), row.getAs[String]("item_id")), row.getAs[Int]("count")))
        //        println(finalBuyData.count()) // 5976

        // 7. 计算概率，即在每次加购物车的小时中，成功购买的次数/当前小时中加购物车的次数

        // 7.1 join:
        // 先通过join操作，找到于最后一天中所有成功购买的"用户-商品对"
        // lastAddData:((user_id,item_id), lastHour) inner join finalBuyData((user_id,item_id), 1)
        //      => ((user_id,item_id),(lastHour,1))
        // 7.2 map:
        // 将小时作为key，属于该小时内的"用户-商品对"作为value，即：(hour, (user_id,item_id))
        // 7.3 groupByKey:
        // 按小时进行分组，即：(hour, Iterable(user_id, item_id))
        // 7.4 join:
        // 通过join操作，将"之前统计出属于该小时内的加入购物车的操作总数"作为新一列，即：(hour, (Iterable(user_id, item_id), counts))
        // 7.5 map:
        // 计算概率，Iterable(user_id, item_id).size即为距离最后一天前hour小时加入到购物车内的"用户-商品对"的个数，则概率为size/counts
        // 生成格式：(hour，该小时内加入购物车/最后一天成功购买的量)
        // 7.6 sortByKey:
        // 按照hour从小到大排序
        // 7.7 collect:
        // 将map的结果"收集"起来
        lastAddData.join(finalBuyData.rdd)
            .map(row => (row._2._1, row._1))
            .groupByKey()
            .join(countsByHours)
            .map(row => (row._1, (row._2._1.size.toDouble / row._2._2).formatted("%.4f")))
            .sortByKey()
            .collect()
            .toStream.toDS().write.csv("src/main/resources/results/part-demo-3")

    }

    def getBetweenHours(big: Date, small: Date): Long = {
        (big.getTime - small.getTime) / (1000 * 60 * 60)
    }

    /**
      * 获得最后一天增加到购物车中的记录
      */
    def getLastDayData(): Unit = {
        // 1. 筛选出第30天中最后一次用户加购物车操作(3)
        val dayDate30: java.sql.Date = java.sql.Date.valueOf("2014-12-18")

        val lastAdd = df.filter(row => simpleDateFormat.parse(row.getString(5)).after(dayDate30))
            .filter(row => 3.equals(Integer.valueOf(row.getString(2))))
        // 2. 去除多余列及重复数据
        val clearData = lastAdd.drop("user_geohash", "item_category", "behavior_type", "time").distinct()
            .map(row => (row.getAs[String]("user_id"), row.getAs[String]("item_id")))
        // 3. 写入csv
        clearData.collect.toStream.toDS.write.csv("src/main/resources/results/part-demo-2")
    }

}
