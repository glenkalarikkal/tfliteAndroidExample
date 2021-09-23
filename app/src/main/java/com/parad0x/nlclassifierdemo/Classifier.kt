package com.parad0x.nlclassifierdemo

interface Classifier {
    fun classify(text: String): Classification
}

data class Classification(val label: String, val confidence: Float)