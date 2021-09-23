package com.parad0x.nlclassifierdemo

import android.content.Context
import com.parad0x.nlclassifierdemo.ml.SimpleClassifier2
import org.tensorflow.lite.DataType
import org.tensorflow.lite.schema.TensorType.FLOAT32
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer

class SimpleClassifier(private val context: Context) {

    fun classify(byteBuffer: ByteBuffer){
        val model = SimpleClassifier2.newInstance(context)

        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 1000), DataType.FLOAT32)
        inputFeature0.loadBuffer(byteBuffer)

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        // Releases model resources if no longer used.
        model.close()
    }
}