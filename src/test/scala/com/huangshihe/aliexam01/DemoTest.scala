package com.huangshihe.aliexam01

import java.text.{DateFormat, SimpleDateFormat}
import java.util.Date

import org.junit.Test

/**
  * Created by huangshihe on 28/03/2017.
  */
@Test
class DemoTest {

    @Test
    def lastTime3() = {
        Demo.lastTime3()
    }

    @Test
    def getBetweenHours() = {
        val simpleDateFormat: DateFormat = new SimpleDateFormat("yyyy-MM-dd HH")
        val bigDate = simpleDateFormat.parse("2014-11-19 13")
        val smallDate = simpleDateFormat.parse("2014-11-18 12")
        println(Demo.getBetweenHours(bigDate, smallDate))
    }

}