package com.huangshihe.aliexam01

import java.io.PrintWriter

import scala.io.Source

/**
  * Created by huangshihe on 31/03/2017.
  */
object Utils {
    def formatJson(): Unit = {
        val file = Source.fromFile("src/main/show/results.csv")
        val printWriter = new PrintWriter("src/main/show/results_1.json")
        val stringBuilder = new StringBuffer("[")
        file.getLines().foreach(line => {
            stringBuilder.append("[")
            stringBuilder.append(line)
            stringBuilder.append("],")
        })
        if (stringBuilder.length() > 1) {
            stringBuilder.replace(stringBuilder.length() - 1, stringBuilder.length(), "]")
        } else {
            stringBuilder.append("]")
        }
        printWriter.append(stringBuilder.toString)
        printWriter.flush()
        printWriter.close()
    }

}
