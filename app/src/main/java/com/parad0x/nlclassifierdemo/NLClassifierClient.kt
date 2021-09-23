package com.parad0x.nlclassifierdemo

import android.content.Context
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.text.nlclassifier.NLClassifier

const val MODEL_PATH = "model_float16.tflite"
class NLClassifierClient(private val context: Context): Classifier {
    private var classifier: NLClassifier? = null

    fun load(){
        try {
            classifier = NLClassifier.createFromFile(context, MODEL_PATH);
        }catch (e: Exception){
            println(e)
        }
    }

    fun unload(){
        classifier?.close()
        classifier = null
    }

    override fun classify(text: String): Classification {
        val apiResults: List<Category> = classifier!!.classify(text)
        val results: MutableList<Result> = ArrayList(apiResults.size)
        for (i in apiResults.indices) {
            val category: Category = apiResults[i]
            results.add(Result("" + i, category.label, category.score))
        }
        results.sortByDescending { it.score }
        return Classification(results[0].label, results[0].score)
    }
}

class Result(val index: String, val label: String, val score: Float): Comparable<Result> {
    override fun compareTo(other: Result): Int {
        return other.score.compareTo(this.score)
    }
}