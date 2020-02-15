package com.example.Bom;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.AsyncTask;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

public class TensorFlowYoloDetector implements Classifier {
    private long start;
    private long end;
    private static final double[] ANCHORS = { 1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071 };
    private static final int MAX_RESULTS = 5;
    private static final int NUM_CLASSES = 6;
    private static final int NUM_BOXES_PER_BLOCK = 5;
    private static final String[] LABELS = {
            "Neoguri",
            "ShinRamyun",
            "Kkongchi",
            "Godeungeo",
            "Kkaennip",
            "Myeongi"
    };

    private String inputName;
    private int inputSize;

    private int[] intValues;
    private float[] floatValues;
    private String outputName;
    private int blockSize;
    private boolean logStats = false;
    private TensorFlowInferenceInterface inferenceInterface;
    private int gridWidth = 13;
    private int gridHeight = 13;
    private float[] output = new float[9295];
    private PriorityQueue<Recognition> pq = new PriorityQueue<Recognition>(1,
            new SortRecognitionViaPriority());
    private final float[] classes = new float[NUM_CLASSES];

    /** Initializes a native TensorFlow session for classifying images. */
    public static Classifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final int inputSize,
            final String inputName,
            final String outputName,
            final int blockSize) {
        TensorFlowYoloDetector d = new TensorFlowYoloDetector();
        d.inputName = inputName;
        d.inputSize = inputSize;

        d.outputName = outputName;
        d.intValues = new int[inputSize * inputSize];
        d.floatValues = new float[inputSize * inputSize * 3];
        d.blockSize = blockSize;

        d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        return d;
    }

    private TensorFlowYoloDetector() {}

    private float expit(final float x) {
        return (float) (1. / (1. + Math.exp(-x)));
    }

    private void softmax(final float[] vals) {
        float max = Float.NEGATIVE_INFINITY;
        for (final float val : vals) {
            max = Math.max(max, val);
        }
        float sum = 0.0f;
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = (float) Math.exp(vals[i] - max);
            sum += vals[i];
        }
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = vals[i] / sum;
        }
    }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            floatValues[i * 3 + 0] = ((intValues[i] >> 16) & 0xFF) / 255.0f;
            floatValues[i * 3 + 1] = ((intValues[i] >> 8) & 0xFF) / 255.0f;
            floatValues[i * 3 + 2] = (intValues[i] & 0xFF) / 255.0f;
        }

        new TIIAsync().execute();

        for (int y = 0; y < gridHeight; ++y) {
            for (int x = 0; x < gridWidth; ++x) {
                for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
                    final int offset =
                            (gridWidth * (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5))) * y
                                    + (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5)) * x
                                    + (NUM_CLASSES + 5) * b;

                    final float xPos = (x + expit(output[offset + 0])) * blockSize;

                    final float w = (float) (Math.exp(output[offset + 2]) * ANCHORS[2 * b + 0]) * blockSize;
                    final float realXpos = Math.max(0, xPos - w / 2);

                    final float confidence = expit(output[offset + 4]);

                    int detectedClass = -1;
                    float maxClass = 0;

                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        classes[c] = output[offset + 5 + c];
                    }
                    softmax(classes);

                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        if (classes[c] > maxClass) {
                            detectedClass = c;
                            maxClass = classes[c];
                        }
                    }

                    final float confidenceInClass = maxClass * confidence;
                    if (confidenceInClass > 0.01) {
                        pq.add(new Recognition("" + offset, LABELS[detectedClass], confidenceInClass, detectedClass, realXpos));
                    }
                }
            }
        }

        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
            recognitions.add(pq.poll());
        }

        return recognitions;
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }

    class SortRecognitionViaPriority implements Comparator<Recognition> {
    @Override
    public int compare(final Recognition lhs, final Recognition rhs) {
        return Float.compare(rhs.getConfidence(), lhs.getConfidence());
        }
    }

    class TIIAsync extends AsyncTask<Void, Void, Void> {

        @Override
        protected Void doInBackground(Void... voids) {

            start = System.currentTimeMillis();
            inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);
            inferenceInterface.run(new String[]{outputName}, logStats);
            inferenceInterface.fetch(outputName, output);
            end = System.currentTimeMillis();

            return null;
        }
    }


}
