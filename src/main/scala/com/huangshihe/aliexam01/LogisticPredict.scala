package com.huangshihe.aliexam01

import java.text.{DateFormat, SimpleDateFormat}
import java.util.Date

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql._

/**
  * Created by huangshihe on 05/05/2017.
  */
object LogisticPredict {

    Logger.getLogger("org").setLevel(Level.ERROR)
    //        spark.sparkContext.setLogLevel("error")

    val spark: SparkSession = SparkSession.builder().appName("Tianmao logistic predict").master("local")
//        .config("spark.executor.memory", "3g").config("spark.driver.memory", "2g")
        .getOrCreate()
    //    Exception in thread "main" java.lang.OutOfMemoryError: GC overhead limit exceeded


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
      * (type_1即为否浏览过；type_2为是否收藏；type_3为是否加入购物车；type_4为是否买过)
      * 数据输入列：user_id,item_id,time1,time2,time3,type_1,type_2,type_3,type_4
      * 自变量：time1,time2,time3,type_1count,type_2count,type_3count,bought
      * 因变量：type_4(为在最后一天是否买过)
      *
      * ----------------------
      * [TODO 问题：增加自变量失败] step 2
      * 自变量增加type_1count,type_2count,type_3count (即为浏览过的次数，收藏的次数，加入购物车的次数)
      * 自变量减少type_1，type_1为是否浏览，为虚拟变量，可以用type_2和type_3替代
      * 因为对于一条记录：type_1 + type_2 + type_3 == 1
      * （也就是说其实只有两个变量type_2,type_3，第三个为虚拟变量type_1，可以用其他两个变量代替如：1 - type_2 - type_3，
      * 那么就不能加入第三个变量，除非打破上述规则：三个变量之和为1）
      * 变量之间不能有线性关系，即不能有"多重共线"
      * ----------------------
      * step 3
      * 考虑是否有type_1和type_1count同时存在的必要性，或者只用哪一个更好？同理如行为类型2、3。使用模型验证。
      * 当前遇到的2个问题：
      * 1. 时间如何保留，比如有这一种情况，一个人浏览了一件商品很多次，那么把第一次浏览的时间作为自变量还是把最后一次浏览的时间作为自变量？
      * 还是把所有浏览时间的总和（好像没有意义）
      * TODO 实现如下思路：
      * 思路=> 逐步回归：有关系的自变量都放进去：第一次浏览的时间，或者也可以加入：最后一天中，总操作次数/总操作耗时。需要进一步考虑决策周期？
      * 逐步回归参考：http://www.cnblogs.com/shishanyuan/p/4699644.html
      *
      * 2. 如何预测？用30天的数据建立好模型后，要预测第31天的行为，那么用于测试的数据是把前30天的数据再次作为输入，输入到模型中，输出预测值，
      * 还是怎么办？或是把前29天的数据作为训练模型的数据，把第30天的数据作为输入？然后输出第30天会购买的数据？
      *
      * 在输入数据时，要将购买时间置为第31天，即31天的1——24小时内。
      * 也就是说训练模型时，也需要将购买时间作为自变量？那么如果购买多次呢？保留最后一个时间？
      * TODO （重要）实现如下思路
      * 思路=> 用1-29天建立模型，增加自变量：1-29天内有没有买过（买过多少次），第30天(最后一天)买还是没买作为因变量。
      * 思路=> 不需要加入购买时间作为自变量，因为因变量是最后一天买没买，买了就说明，购买时间为最后一天，也就是说购买时间已经体现在模型中。
      * 思路=> 将1-30天的数据作为输入，输入到模型中，输出第31天预测会购买的记录。（对应的时间需要修改，所有用的时间都是距离最后一天还剩余的小时数，那么这里的最后一天就是第31天，而不是第30天）
      *
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
        //        // 将数据转换格式
        //        // ref: http://stackoverflow.com/questions/29383107/how-to-change-column-types-in-spark-sqls-dataframe
        //        val wholeData = df.withColumn("user_id", df("user_id").cast(IntegerType))
        //            .withColumn("item_id", df("item_id").cast(IntegerType))
        //            .withColumn("behavior_type", df("behavior_type").cast(IntegerType))
        //            .withColumn("time", getDiffHour(df("time")))
        //            .drop("user_geohash", "item_category") // TODO 暂时不考虑"用户位置"和"商品分类标识"
        //
        //        //        wholeData.take(10).foreach(println)
        //        // TODO 将数据按时间倒序
        //        // 将数据按行为类型分成4类
        //        // 浏览过的统计数据，生成的结构：user_id,item_id,time,type_1count
        //        val dataOfView = wholeData.filter(row => row.getAs[Int]("behavior_type") == VIEW_TYPE).drop("behavior_type")
        //            .groupBy("user_id", "item_id").agg(min("time").as("time1"), count("time").cast(IntegerType).as("type_1count"))
        //        // 收藏的记录：user_id,item_id,time
        //        val dataOfCollect = wholeData.filter(row => row.getAs[Int]("behavior_type") == COLLECT_TYPE).drop("behavior_type")
        //            .groupBy("user_id", "item_id").agg(min("time").as("time2"), count("time").cast(IntegerType).as("type_2count"))
        //        // 加购物车的记录：user_id,item_id,time
        //        val dataOfAdd = wholeData.filter(row => row.getAs[Int]("behavior_type") == ADD_TO_CART_TYPE).drop("behavior_type")
        //            .groupBy("user_id", "item_id").agg(min("time").as("time3"), count("time").cast(IntegerType).as("type_3count"))
        //        // 成功购买的记录：user_id,item_id,time,label=1.0
        //        val dataOfBuy = wholeData.filter(row => row.getAs[Int]("behavior_type") == BUY_TYPE).drop("time")
        //            .drop("behavior_type").withColumn("label", functions.lit(1.0))
        //
        //        // (user_id,item_id,time1,type_1count)join(user_id,item_id,time2,type2_count)join(user_id,item_id,time3,type3_count)
        //        // => (user_id,item_id,time1,time2,time3,type_1count,type_2count,type_3count)
        //        // TODO 这里的时间应该选取：最后一次type操作时间？第一次type操作时间？
        //        // 讲道理来说：收藏和加购物车的记录之前都应该有对应的浏览记录，因此可以使用左连接，保证数据都是全的。但是这里还是使用全连接，比较稳妥
        //        //        println("view:" + dataOfView.count()) // 8840964
        //        //        println("collect:" + dataOfCollect.count()) // 432391
        //        //        println("add:" + dataOfAdd.count()) // 542806
        //
        //        val all = dataOfView.join(dataOfCollect, Seq("user_id", "item_id"), "outer").join(dataOfAdd, Seq("user_id", "item_id"), "outer")
        //        // 使用when和otherwise，将null转为0
        //        val allWithOutNull = all.select(all("user_id"), all("item_id")
        //            , when(all("time1").isNull, 0).otherwise(all("time1")).as("time1")
        //            , when(all("type_1count").isNull, 0).otherwise(all("type_1count")).as("type_1count")
        //            , when(all("time2").isNull, 0).otherwise(all("time2")).as("time2")
        //            , when(all("type_2count").isNull, 0).otherwise(all("type_2count")).as("type_2count")
        //            , when(all("time3").isNull, 0).otherwise(all("time3")).as("time3")
        //            , when(all("type_3count").isNull, 0).otherwise(all("type_3count")).as("type_3count"))
        //
        //        //        println("all:" + allWithOutNull.count()) // inner join:36610 // outer join with null:8851499 // outer join without null:8851499
        //        //        all.printSchema()
        //        //        root
        //        //        |-- user_id: integer (nullable = true)
        //        //        |-- item_id: integer (nullable = true)
        //        //        |-- time1: integer (nullable = true)
        //        //        |-- type_1count: integer (nullable = false)
        //        //        |-- time2: integer (nullable = true)
        //        //        |-- type_2count: integer (nullable = false)
        //        //        |-- time3: integer (nullable = true)
        //        //        |-- type_3count: integer (nullable = false)
        //
        //        // (user_id,item_id,time,type_1count...) join (user_id,item_id,label) by (user_id,item_id)
        //        // => (user_id,item_id,time,type_1count...,label=1.0)
        //        // join（通过(user_id,item_id)进行join操作）已成功购买的记录，加上label=1.0
        //        val positiveData = allWithOutNull.join(dataOfBuy, Seq("user_id", "item_id"))
        //        // 所有的记录与标记为1.0的浏览的记录的差集即为浏览了但是没有生成购买记录的记录即未购买的记录
        //        val negativeData = allWithOutNull.except(positiveData.drop("label")).withColumn("label", functions.lit(0.0))
        //
        //        val data =

        //        data.take(50).foreach(println)
        //        [3349105,396219180,492,2,0,0,492,1,1.0]
        //        [3436360,275074350,14,26,0,0,168,3,1.0]
        //        [3436360,275074350,14,26,0,0,168,3,1.0]
        //        [3632806,349045655,148,16,0,0,609,2,1.0]
        //        [3638392,109432312,655,2,0,0,0,0,1.0]
        //        [3681631,128590725,736,5,0,0,0,0,1.0]
        //        [3773095,213873016,156,4,0,0,156,1,1.0]
        //        [3883093,174457660,213,2,0,0,213,2,1.0]
        //        [3974224,172523395,651,11,0,0,0,0,1.0]

        // TODO 使用foreach，分布式使用foreach难度在哪？

        // 生成特征值
        val vectorAssembler = new VectorAssembler()
            .setInputCols(Array("time1", "type_1count", "time2", "type_2count", "time3", "type_3count", "type_4count"))
            .setOutputCol("features")

        // 使用逻辑回归
        val lor = new LogisticRegression()
        // 新建pipeline，分为三个阶段：分词、转换特征向量和逻辑回归
        val pipeline = new Pipeline().setStages(Array(vectorAssembler, lor))
        // 训练模型
        val model = pipeline.fit(getTrainingData)

        //        import spark.implicits._
        //        val test = Seq((10001082, 53616768, 1, 5, 3, 1, 1, 1, 0)
        //            , (10001082, 53616768, 10, 50, 30, 1, 0, 0, 1)
        //            , (10001082, 53616768, 10, 50, 30, 1, 0, 0, 0)
        //            , (10001082, 53616768, 1, 50, 30, 1, 0, 0, 1)
        //            , (10001082, 53616768, 1, 50, 1, 1, 0, 0, 0)
        //            , (10001082, 53616768, 10, 3, 5, 1, 1, 1, 0)
        //            , (10001082, 53616768, 1, 1, 1, 1, 1, 1, 0)
        //            , (10001082, 290088061, 10, 5, 3, 2, 1, 2, 0)
        //            , (10001082, 323339743, 0, 0, 0, 0, 0, 0, 0)
        //        ).toDF("user_id", "item_id", "time1", "type_1count", "time2", "type_2count", "time3", "type_3count", "type_4count")
        //
        //        model.transform(test).show()

        import spark.implicits._

        model.transform(getInputData)
            .filter(row => row.getAs[Double]("prediction") >= 0)
            .map(row => (row.getAs[Int]("user_id"), row.getAs[Int]("item_id")))
            .collect()
            .toStream.toDS().write.csv("src/main/resources/results/part-demo-5")
    }

    /**
      * 获得1-29天的数据作为训练数据，并记录1-29天内成功购买的次数。
      * label为1即为第30天购买了的记录，否则label为0
      *
      * @return
      */
    def getTrainingData: DataFrame = {
        // 将数据转换格式 ref: http://stackoverflow.com/questions/29383107/how-to-change-column-types-in-spark-sqls-dataframe
        val wholeTrainingData = df
            .drop("user_geohash", "item_category") // TODO 暂时不考虑"用户位置"和"商品分类标识"
            .filter(row => simpleDateFormat.parse(row.getAs[String]("time")).before(day30))
            .withColumn("user_id", df("user_id").cast(IntegerType))
            .withColumn("item_id", df("item_id").cast(IntegerType))
            .withColumn("behavior_type", df("behavior_type").cast(IntegerType))
            .withColumn("time", getTrainHour(df("time")))

        // 将数据按行为类型分成4类
        // 浏览过的统计数据，生成的结构：user_id,item_id,time1,type_1count
        val dataOfView = wholeTrainingData.filter(row => row.getAs[Int]("behavior_type") == VIEW_TYPE).drop("behavior_type")
            .groupBy("user_id", "item_id").agg(min("time").as("time1"), count("time").cast(IntegerType).as("type_1count"))
        // 收藏的记录：user_id,item_id,time2,type_2count
        val dataOfCollect = wholeTrainingData.filter(row => row.getAs[Int]("behavior_type") == COLLECT_TYPE).drop("behavior_type")
            .groupBy("user_id", "item_id").agg(min("time").as("time2"), count("time").cast(IntegerType).as("type_2count"))
        // 加购物车的记录：user_id,item_id,time3,time_3count
        val dataOfAdd = wholeTrainingData.filter(row => row.getAs[Int]("behavior_type") == ADD_TO_CART_TYPE).drop("behavior_type")
            .groupBy("user_id", "item_id").agg(min("time").as("time3"), count("time").cast(IntegerType).as("type_3count"))
        // 成功购买的记录：user_id,item_id,time_4count
        val dataOfBuy = wholeTrainingData.filter(row => row.getAs[Int]("behavior_type") == BUY_TYPE).drop("behavior_type")
            .groupBy("user_id", "item_id").agg(count("time").cast(IntegerType).as("type_4count"))

        // (user_id,item_id,time1,type_1count)join(user_id,item_id,time2,type2_count)
        // join(user_id,item_id,time3,type3_count)join(user_id,item_id,type4_count)
        // => (user_id,item_id,time1,time2,time3,type_1count,type_2count,type_3count,type_4count)
        // TODO 这里的时间应该选取：最后一次type操作时间？第一次type操作时间？或者两个时间都加上，但是都加上不一定可以提高准确率

        // 讲道理来说：收藏和加购物车的记录之前都应该有对应的浏览记录，因此可以使用左连接，保证数据都是全的。但是这里还是使用全连接，比较稳妥
        val all = dataOfView.join(dataOfCollect, Seq("user_id", "item_id"), "outer")
            .join(dataOfAdd, Seq("user_id", "item_id"), "outer")
            .join(dataOfBuy, Seq("user_id", "item_id"), "outer")
        // 使用when和otherwise，将null转为0
        val allWithOutNull = all.select(all("user_id"), all("item_id")
            , when(all("time1").isNull, 0).otherwise(all("time1")).as("time1")
            , when(all("type_1count").isNull, 0).otherwise(all("type_1count")).as("type_1count")
            , when(all("time2").isNull, 0).otherwise(all("time2")).as("time2")
            , when(all("type_2count").isNull, 0).otherwise(all("type_2count")).as("type_2count")
            , when(all("time3").isNull, 0).otherwise(all("time3")).as("time3")
            , when(all("type_3count").isNull, 0).otherwise(all("type_3count")).as("type_3count")
            , when(all("type_4count").isNull, 0).otherwise(all("type_4count")).as("type_4count")
        )
        // 因为df中只有1-30天的数据，所以超过30天0点的数据即为第30天的数据
        // 先进行投影、选择，drop掉不需要的列，减少数据量
        val lastDayBuy = df.drop("user_geohash", "item_category")
            .withColumn("user_id", df("user_id").cast(IntegerType))
            .withColumn("item_id", df("item_id").cast(IntegerType))
            .withColumn("behavior_type", df("behavior_type").cast(IntegerType))
            .filter(row => row.getAs[Int]("behavior_type") == BUY_TYPE).drop("behavior_type")
            .filter(row => simpleDateFormat.parse(row.getAs[String]("time")).after(day30)).drop("time")
            .withColumn("label", functions.lit(1.0))

        // 最后一天购买的量确实相当少
        //        println("last day buy count:" + lastDayBuy.count()) //6765

        // (user_id,item_id,time1,type_1count...) join (user_id,item_id,label) by (user_id,item_id)
        // => (user_id,item_id,time1,type_1count...,label=1.0)
        // join（通过(user_id,item_id)进行join操作）已成功购买的记录，加上label=1.0
        val positiveData = allWithOutNull.join(lastDayBuy, Seq("user_id", "item_id"))
        // 所有的记录与标记为1.0的浏览的记录的差集即为浏览了但是没有生成购买记录的记录即未购买的记录
        val negativeData = allWithOutNull.except(positiveData.drop("label")).withColumn("label", functions.lit(0.0))

        positiveData union negativeData
    }

    /**
      * 获得模型的输入数据，即获得1-30天的数据作为模型输入，其中最后一天为第31天，也就是时间基准为第31天
      *
      * @return
      */
    def getInputData: DataFrame = {
        val wholeTrainingData = df
            .drop("user_geohash", "item_category") // TODO 暂时不考虑"用户位置"和"商品分类标识"
            .withColumn("user_id", df("user_id").cast(IntegerType))
            .withColumn("item_id", df("item_id").cast(IntegerType))
            .withColumn("behavior_type", df("behavior_type").cast(IntegerType))
            .withColumn("time", getInputHour(df("time")))

        // 将数据按行为类型分成4类
        // 浏览过的统计数据，生成的结构：user_id,item_id,time1,type_1count
        val dataOfView = wholeTrainingData.filter(row => row.getAs[Int]("behavior_type") == VIEW_TYPE).drop("behavior_type")
            .groupBy("user_id", "item_id").agg(min("time").as("time1"), count("time").cast(IntegerType).as("type_1count"))
        // 收藏的记录：user_id,item_id,time2,type_2count
        val dataOfCollect = wholeTrainingData.filter(row => row.getAs[Int]("behavior_type") == COLLECT_TYPE).drop("behavior_type")
            .groupBy("user_id", "item_id").agg(min("time").as("time2"), count("time").cast(IntegerType).as("type_2count"))
        // 加购物车的记录：user_id,item_id,time3,time_3count
        val dataOfAdd = wholeTrainingData.filter(row => row.getAs[Int]("behavior_type") == ADD_TO_CART_TYPE).drop("behavior_type")
            .groupBy("user_id", "item_id").agg(min("time").as("time3"), count("time").cast(IntegerType).as("type_3count"))
        // 成功购买的记录：user_id,item_id,time_4count
        val dataOfBuy = wholeTrainingData.filter(row => row.getAs[Int]("behavior_type") == BUY_TYPE).drop("behavior_type")
            .groupBy("user_id", "item_id").agg(count("time").cast(IntegerType).as("type_4count"))

        // 讲道理来说：收藏和加购物车的记录之前都应该有对应的浏览记录，因此可以使用左连接，保证数据都是全的。但是这里还是使用全连接，比较稳妥
        val all = dataOfView.join(dataOfCollect, Seq("user_id", "item_id"), "outer")
            .join(dataOfAdd, Seq("user_id", "item_id"), "outer")
            .join(dataOfBuy, Seq("user_id", "item_id"), "outer")
        // 使用when和otherwise，将null转为0
        all.select(all("user_id"), all("item_id")
            , when(all("time1").isNull, 0).otherwise(all("time1")).as("time1")
            , when(all("type_1count").isNull, 0).otherwise(all("type_1count")).as("type_1count")
            , when(all("time2").isNull, 0).otherwise(all("time2")).as("time2")
            , when(all("type_2count").isNull, 0).otherwise(all("type_2count")).as("type_2count")
            , when(all("time3").isNull, 0).otherwise(all("time3")).as("time3")
            , when(all("type_3count").isNull, 0).otherwise(all("type_3count")).as("type_3count")
            , when(all("type_4count").isNull, 0).otherwise(all("type_4count")).as("type_4count")
        )
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

    @Deprecated
    // 这里的参数time，不能使用_代替，否则报错
    val getDiffHour: UserDefinedFunction =
        udf((time: String) => ((day31.getTime - simpleDateFormat.parse(time).getTime) / (1000 * 60 * 60)).toInt)

    // 获得训练数据的时间差，即用1-29天建模，因变量第30天会不会购买，时间差基准为第30天
    val getTrainHour: UserDefinedFunction =
        udf((time: String) => ((day30.getTime - simpleDateFormat.parse(time).getTime) / (1000 * 60 * 60)).toInt)

    // 获得输入数据的时间差，时间差基准为第31天
    val getInputHour: UserDefinedFunction =
        udf((time: String) => ((day31.getTime - simpleDateFormat.parse(time).getTime) / (1000 * 60 * 60)).toInt)

}
