package com.example.Bom;

import android.graphics.Bitmap;

import java.util.List;

public interface Classifier {

    class Recognition {
        private final String id;
        private final String title;
        private final Float confidence;
        private final int detectedClass;
        private final float xPos;

        public Recognition(
                final String id, final String title, final Float confidence, final int detectedClass, final float xPos) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.detectedClass = detectedClass;
            this.xPos = xPos;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public int getDetectedClass() {
            return detectedClass;
        }

        public float getXpos() {
            return xPos;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            return resultString.trim();
        }
    }

    List<Recognition> recognizeImage(Bitmap bitmap);

    void close();
}