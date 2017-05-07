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
      * 模型思路：
      * 根据行为类型将数据分为四类，1-3类中的记录若在4中出现，即成功购买的标注1，否则标注0。
      * 数据输入列：user_id,item_id,time,oper_1,oper_2,oper_3 (oper_1即为否浏览过；oper_2为是否收藏；oper_3为是否加入购物车)
      * 自变量：time,oper_1,oper_2,oper_3
      * 因变量：oper_4
      *
      * ----------------------
      * TODO step 2
      * 自变量增加oper_1count,oper_2count,oper_3count (即为浏览过的次数，收藏的次数，加入购物车的次数)
      * ----------------------
      * TODO step 3
      * 考虑是否有oper_1和oper_1count同时存在的必要性，或者只用哪一个更好？同理如行为类型2、3。使用模型验证。
      * ----------------------
      * TODO step 4
      * 自变量加入用户位置的空间标识
      * ----------------------
      * TODO step 5
      * 自变量加入商品位置的空间标识
      * ----------------------
      * TODO step 6
      * 自变量加入商品分类标识
      *
      * ----------------------
      * 进阶（难度无先后顺序）：
      * 1. 浏览操作和收藏、加购物车、购买的相关性？如是否收藏操作之前肯定有浏览记录？（即自变量之间存在相关性，而这是要尽量避免的）
      * 2. 多次收藏的含义？多次加购物车的含义？
      * 3. 使用逐步回归？
      * 4. 使用蒙特卡洛模拟？
      * 5. 使用神经网络？
      * 6. 使用决策树？
      * 7. 使用随机森林？
      * 8. 集成多个模型的结果？集成算法？提升多少？需验证。
      * 9. 修改阈值？专家打分法？为了提高准确率（牺牲其他指标）？
      *
      */
    def getPredictData(): Unit = {
        // 将数据转换格式
        // ref: http://stackoverflow.com/questions/29383107/how-to-change-column-types-in-spark-sqls-dataframe
        val wholeData = df.withColumn("user_id", df("user_id").cast(IntegerType))
            .withColumn("item_id", df("item_id").cast(IntegerType))
            .withColumn("behavior_type", df("behavior_type").cast(IntegerType))
            .withColumn("time", getDiffHour(df("time")))
            .drop("user_geohash", "item_category") // TODO 暂时不考虑"用户位置"和"商品分类标识"

        wholeData.take(10).foreach(println)

        // 将数据按行为类型分成4类
        val dataOfView = wholeData.filter(row => row.getAs[Int]("behavior_type") == VIEW_TYPE)
        val dataOfCollect = wholeData.filter(row => row.getAs[Int]("behavior_type") == COLLECT_TYPE)
        val dataOfAdd = wholeData.filter(row => row.getAs[Int]("behavior_type") == ADD_TO_CART_TYPE)
        val dataOfBuy = wholeData.filter(row => row.getAs[Int]("behavior_type") == BUY_TYPE)

        // 预处理后，将1-3类数据合并


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
        udf((time: String) => ((day31.getTime - simpleDateFormat.parse(time).getTime) / (1000 * 60 * 60)).toInt)

}
