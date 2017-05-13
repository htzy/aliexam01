package com.huangshihe.aliexam01

import java.text.{DateFormat, SimpleDateFormat}
import java.util.Date

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession, functions}

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
      * 数据输入列：user_id,item_id,time,type_1,type_2,type_3 (type_1即为否浏览过；type_2为是否收藏；type_3为是否加入购物车)
      * 自变量：time,type_1count,type_2,type_2count,type_3,type_3count
      * 因变量：type_4
      *
      * ----------------------
      * step 2
      * 自变量增加type_1count,type_2count,type_3count (即为浏览过的次数，收藏的次数，加入购物车的次数)
      * 自变量减少type_1，type_1为是否浏览，为虚拟变量，可以用type_2和type_3替代
      * 因为对于一条记录：type_1 + type_2 + type_3 == 1
      * （也就是说其实只有两个变量type_2,type_3，第三个为虚拟变量type_1，可以用其他两个变量代替如：1 - type_2 - type_3，
      * 那么就不能加入第三个变量，除非打破上述规则：三个变量之和为1）
      * 变量之间不能有线性关系，即不能有"多重共线"
      * ----------------------
      * TODO step 3
      * 考虑是否有type_1和type_1count同时存在的必要性，或者只用哪一个更好？同理如行为类型2、3。使用模型验证。
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

        //        wholeData.take(10).foreach(println)
        // TODO 将数据按时间倒序
        // 将数据按行为类型分成4类
        // 浏览的记录：user_id,item_id,time
        val dataOfView = wholeData.filter(row => row.getAs[Int]("behavior_type") == VIEW_TYPE).drop("behavior_type")
        // 收藏的记录：user_id,item_id,time
        val dataOfCollect = wholeData.filter(row => row.getAs[Int]("behavior_type") == COLLECT_TYPE).drop("behavior_type")
        // 加购物车的记录：user_id,item_id,time
        val dataOfAdd = wholeData.filter(row => row.getAs[Int]("behavior_type") == ADD_TO_CART_TYPE).drop("behavior_type")
        // 成功购买的记录：user_id,item_id,time,label=1.0
        val dataOfBuy = wholeData.filter(row => row.getAs[Int]("behavior_type") == BUY_TYPE)
            .drop("behavior_type").withColumn("label", functions.lit(1.0))

        // join（通过(user_id,item_id,time)进行join操作）已成功购买的记录，加上label=1.0
        val positiveView = dataOfView.join(dataOfBuy, Seq("user_id", "item_id", "time"))
        // 实践证明：不要随便用差集，贼慢！
        // 浏览的记录与标记为1.0的浏览的记录的差集即为浏览了但是没有生成购买记录的记录
        val negativeView = dataOfView.except(positiveView.drop("label")).withColumn("label", functions.lit(0.0))
        val allView = positiveView.union(negativeView)
            .withColumn("type_1", functions.lit(true))
            .withColumn("type_2", functions.lit(false)).withColumn("type_3", functions.lit(false))

        val positiveCollect = dataOfCollect.join(dataOfBuy, Seq("user_id", "item_id", "time"))
        val negativeCollect = dataOfCollect.except(positiveCollect.drop("label")).withColumn("label", functions.lit(0.0))
        val allCollect = positiveCollect.union(negativeCollect)
            .withColumn("type_2", functions.lit(true))
            .withColumn("type_1", functions.lit(false)).withColumn("type_3", functions.lit(false))

        val positiveAdd = dataOfAdd.join(dataOfBuy, Seq("user_id", "item_id", "time"))
        val negativeAdd = dataOfAdd.except(positiveAdd.drop("label")).withColumn("label", functions.lit(0.0))
        val allAdd = positiveAdd.union(negativeAdd)
            .withColumn("type_3", functions.lit(true))
            .withColumn("type_1", functions.lit(false)).withColumn("type_2", functions.lit(false))

        // 预处理后，将1-3类数据合并
        val data = allView union allCollect union allAdd

        //        data.show(20)

        // 生成特征值
        val vectorAssembler = new VectorAssembler().setInputCols(Array("time", "type_1", "type_2", "type_3"))
            .setOutputCol("features")

        //        val features = vectorAssembler.transform(data)
        //        features.show(10)

        // 使用逻辑回归
        val lor = new LogisticRegression()
        // 新建pipeline，分为三个阶段：分词、转换特征向量和逻辑回归
        val pipeline = new Pipeline().setStages(Array(vectorAssembler, lor))
        // 训练模型
        val model = pipeline.fit(data)

        import spark.implicits._
        val test = Seq((10001082, 53616768, 1, 0, 0, 1)
            , (10001082, 53616768, 1, 1, 0, 0)
            , (10001082, 53616768, 1, 0, 1, 0)
            , (10001082, 53616768, 2, 0, 0, 1)
            , (10001082, 290088061, 1, 1, 0, 0)
            , (10001082, 323339743, 1, 0, 1, 0)
        ).toDF("user_id", "item_id", "time", "type_1", "type_2", "type_3")
        model.transform(test).show()

        //        +--------+---------+----+------+------+------+-----------------+--------------------+--------------------+----------+
        //        | user_id|  item_id|time|type_1|type_2|type_3|         features|       rawPrediction|         probability|prediction|
        //        +--------+---------+----+------+------+------+-----------------+--------------------+--------------------+----------+
        //        |10001082| 53616768|   1|     0|     0|     1|[1.0,0.0,0.0,1.0]|[2.54039847655302...|[0.92692582171709...|       0.0|
        //        |10001082| 53616768|   1|     1|     0|     0|[1.0,1.0,0.0,0.0]|[2.54039847655302...|[0.92692582171709...|       0.0|
        //        |10001082| 53616768|   1|     0|     1|     0|[1.0,0.0,1.0,0.0]|[2.54039847655302...|[0.92692582171709...|       0.0|
        //        |10001082| 53616768|   2|     0|     0|     1|[2.0,0.0,0.0,1.0]|[2.54062936941713...|[0.92694145955193...|       0.0|
        //        |10001082|290088061|   1|     1|     0|     0|[1.0,1.0,0.0,0.0]|[2.54039847655302...|[0.92692582171709...|       0.0|
        //        |10001082|323339743|   1|     0|     1|     0|[1.0,0.0,1.0,0.0]|[2.54039847655302...|[0.92692582171709...|       0.0|
        //        +--------+---------+----+------+------+------+-----------------+--------------------+--------------------+----------+

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
