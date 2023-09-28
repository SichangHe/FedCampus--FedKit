package org.eu.fedcampus.fed_kit_examples.mnist

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.eu.fedcampus.fed_kit_train.FlowerClient
import org.eu.fedcampus.fed_kit_train.SampleSpec
import org.eu.fedcampus.fed_kit_train.helpers.classifierAccuracy
import org.eu.fedcampus.fed_kit_train.helpers.maxSquaredErrorLoss
import java.io.File

fun sampleSpec() = SampleSpec<FloatArray, FloatArray>(
    { it.toTypedArray() },
    { it.toTypedArray() },
    { Array(it) { FloatArray(1) } },
    ::maxSquaredErrorLoss,
    ::classifierAccuracy,
)

private suspend fun processSet(dataSetDir: String, call: suspend (Int, String) -> Unit) {
    withContext(Dispatchers.IO) {
        File(dataSetDir).useLines {
            it.forEachIndexed { i, l -> launch { call(i, l) } }
        }
    }
}

suspend fun loadData(
    dataDir: String, flowerClient: FlowerClient<FloatArray, FloatArray>, partitionId: Int
) {
    // proecess training set
    Log.i(TAG, "loading pmdata")
    processSet("$dataDir/p${partitionId.toString().padStart(2, '0')}_train.csv") { index, line ->
        if (index % 100 == 99) Log.i(TAG, "Loading $index th training sample")
        addSample(flowerClient, line, true)
    }
    // process test set
    processSet("$dataDir/test.csv") { index, line ->
        if (index % 100 == 99) Log.i(TAG, "Loading $index th test sample")
        addSample(flowerClient, line, false)
    }
}

private fun addSample(
    flowerClient: FlowerClient<FloatArray, FloatArray>, line: String, isTraining: Boolean
) {

    val splits = line.split(",")
    val label = floatArrayOf(splits.last().toFloat())
    val featureArray = FloatArray(FEATURE_SIZE)
    for (i in featureArray.indices) {
        featureArray[i] = splits[i + 1].toFloat()
    }
    flowerClient.addSample(featureArray, label, isTraining)
}

private const val TAG = "PMData Data Loader"
private const val FEATURE_SIZE = 7
