package com.parad0x.nlclassifierdemo

import android.content.Context
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.MappedByteBuffer
import java.util.Locale

class CustomTFLiteInterpreterClient: Classifier {
    private var tflite: Interpreter? = null
    private val tokenizer = Tokenizer()
    fun load(tokenizerData: String, modelMappedBuffer: MappedByteBuffer) {
        try {
            tflite = Interpreter(modelMappedBuffer)
            tokenizer.load(tokenizerData)
        } catch (e: Exception) {
            println(e)
        }
    }

    fun unload() {
    }

    override fun classify(text: String): Classification {
        val matrix = tokenizer.convertSentencesToMatrix(listOf(text))
        val outputMatrix = Array(10) {FloatArray(1)}
        tflite?.run(matrix, outputMatrix)
        val prediction = outputMatrix[0][0]
        val result = when{
            prediction <= 0.5 -> Classification("Not Insult", 1 - prediction)
            else ->  Classification("Insult", prediction)
        }
        return result
    }
}

class Tokenizer {
    var wordIndices: JSONObject? = null


    fun load(tokenizerData: String) {
        wordIndices = JSONObject(tokenizerData)
    }
    fun tokenizeSentence(sentence: String): List<Int> {
        return sentence.toLowerCase(Locale.ROOT).split(" ").map {
            try {
                wordIndices?.getInt(it) ?: 0
            } catch (e: java.lang.Exception) {
                0
            }
        }
    }

    fun convertSequenceToMatrix(sequences: List<List<Int>>): Array<FloatArray> {
        val lenOfSequences = sequences.size
        val numOfWords = wordIndices?.length()?: 0
        val array = Array(lenOfSequences) { FloatArray(numOfWords) }
        sequences.forEachIndexed { index, sequence ->
            sequence.forEach { wordIndex ->
                array[index][wordIndex] = 1.0f
            }
        }
        return array
    }

    fun convertSentencesToMatrix(sentences: List<String>): Array<FloatArray>{
        return convertSequenceToMatrix(sentences.map { tokenizeSentence(it) })
    }
}