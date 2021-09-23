package com.parad0x.nlclassifierdemo

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.BufferedOutputStream
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

enum class CLASSIFIER_TYPE {
    DEAFULT_NL_CLASSIFIER,
    CUSTOM_TFLITE_CLASSIFIER
}

class MainActivity : AppCompatActivity() {
    private var classifierClient: Classifier? = null
    private var button: Button? = null
    private var textToClassify: EditText? = null
    private var resultView: TextView? = null

    private val classifierType = CLASSIFIER_TYPE.DEAFULT_NL_CLASSIFIER

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        when(classifierType){
            CLASSIFIER_TYPE.DEAFULT_NL_CLASSIFIER -> setupNLClassifierClient()
            CLASSIFIER_TYPE.CUSTOM_TFLITE_CLASSIFIER -> setupCustomClassifierClient()
        }

        button = findViewById(R.id.button)
        textToClassify = findViewById(R.id.textToClassify)
        resultView = findViewById(R.id.resultView)

        button?.setOnClickListener {
            resultView?.visibility = View.INVISIBLE
            val text = textToClassify?.text
            text?.let {
                val results = classifierClient?.classify(it.toString())
                resultView?.text = "Result : ${results?.label}"
                resultView?.visibility = View.VISIBLE
            }
        }

    }

    private fun setupNLClassifierClient() {
        classifierClient = NLClassifierClient(this)
    }

    private fun setupCustomClassifierClient() {
        classifierClient = CustomTFLiteInterpreterClient()

        val modelMappedBuffer = extractModelBuffer()
        val modelVocabulary: String = extractModelVocabulary()

        (classifierClient as CustomTFLiteInterpreterClient).load(modelVocabulary, modelMappedBuffer)
    }

    private fun extractModelVocabulary(): String {
        val data = resources.assets.open("word_indices.json").reader()
        return BufferedReader(data).use { it.readText() }
    }

    private fun extractModelBuffer(): MappedByteBuffer {
        val modelLocation = prepareModelFile()
        val inputStream = FileInputStream(modelLocation)
        val fileChannel: FileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size())
    }

    private fun prepareModelFile(): String {
        val assetManager = assets;
        val configDir = filesDir.absolutePath;
        val stream = assetManager.open("simple_classifier_2.tflite");
        val mTFLiteModelFile = "$configDir/sameple_classifier_2.data";
        val output =  BufferedOutputStream(FileOutputStream(mTFLiteModelFile));
        copyFile(stream, output)
        return mTFLiteModelFile
    }

    override fun onDestroy() {
        super.onDestroy()
        classifierClient = null
    }
}

fun copyFile(ins: InputStream, out: OutputStream) {
    val buffer =  ByteArray(1024)
    var read: Int;
    while(true) {
        read = ins.read(buffer)
        if(read == -1)
            break;
        out.write(buffer, 0, read);
    }
    ins.close()
    out.close()
}