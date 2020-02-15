#include <jni.h>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"
#include <algorithm>
#include <time.h>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define JNICALL

int trainImagesSize;
int cnt = 0;
vector<Mat> trainImages;
vector<KeyPoint> keypoints1;
vector<KeyPoint> keypoints2;
Mat descriptors2;
Mat matImg;
string tts_result;

int knnSize = 0;
double minHessian = 100;
Ptr <SURF> detector = SURF::create();
Ptr <DescriptorMatcher> matcher = DescriptorMatcher::create(
        DescriptorMatcher::FLANNBASED);
vector<vector<DMatch> > knn_matches;
vector<KeyPoint> temp_keypoints;
vector<vector<KeyPoint>> temp_keypoints1;
Mat temp_descriptors;
vector<Mat> temp_descriptors1;
vector<DMatch> temp_good_matches;
vector<DMatch> good_matches;

static void createKeypointsAndDescriptors(const Mat &matInput) {
    int max = 0;
    const float ratio_thresh = 0.75f;
    int result_i = 0;

    detector->setHessianThreshold(minHessian);
    detector->detectAndCompute(matInput, Mat(), keypoints2, descriptors2);

    for (int i = 0; i < trainImagesSize; i++) {
        knn_matches.clear();

        matcher->knnMatch(temp_descriptors1[i], descriptors2, knn_matches, 2);

        knnSize = knn_matches.size();

        for(int j = 0; j < knnSize; j++) {
            if (knn_matches[j][0].distance < ratio_thresh * knn_matches[j][1].distance) {
                temp_good_matches.push_back(knn_matches[j][0]);
            }
        }

        int tgmSize = temp_good_matches.size();

        if (max < tgmSize) {
            max = tgmSize;
            keypoints1 = temp_keypoints1[i];
            matImg = trainImages[i];
            good_matches = temp_good_matches;
            result_i = i;
        }

        temp_good_matches.clear();

        if (max < 60) result_i = 8;

    }

    if(result_i == 0 || result_i == 1) tts_result = "천원";
    else if(result_i == 2 || result_i == 3) tts_result = "오천원";
    else if(result_i == 4 || result_i == 5) tts_result = "만원";
    else if(result_i == 6 || result_i == 7) tts_result = "오만원";
    else if(result_i == 8) tts_result = "";


}


extern "C"
JNIEXPORT void JNICALL
Java_com_example_ocrtest_MainActivity_surfWithFlann(JNIEnv *env, jobject instance,
                                                         jlong matAddrInput,
                                                         jlong image_matches) {
    Mat &matInput = *(Mat *) matAddrInput;
    Mat &imageMatches = *(Mat *) image_matches;

    createKeypointsAndDescriptors(matInput);

    vector<char> mask_matches(good_matches.size(), 0);

    drawMatches(matImg, keypoints1, matInput, keypoints2, good_matches, imageMatches,
                Scalar::all(-1),
                Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
}


extern "C"
JNIEXPORT void JNICALL
Java_com_example_ocrtest_MainActivity_sendImages(JNIEnv *env, jobject instance,
                                                      jlongArray tempAddrObj_) {


    int length = env->GetArrayLength(tempAddrObj_);
    jlong *tempAddrObj = env->GetLongArrayElements(tempAddrObj_, NULL);

    for (int i = 0; i < length; i++) {
        Mat &tempImage = *(Mat *) tempAddrObj[i];
        trainImages.push_back(tempImage);
    }
    env->ReleaseLongArrayElements(tempAddrObj_, tempAddrObj, 0);

    trainImagesSize = trainImages.size();

    for (int i = 0; i < trainImagesSize; i++) {

        detector->detectAndCompute(trainImages[i], Mat(), temp_keypoints, temp_descriptors);
        temp_keypoints1.push_back(temp_keypoints);
        temp_descriptors1.push_back(temp_descriptors);
    }

}

extern "C"
JNIEXPORT jbyteArray JNICALL
Java_com_example_ocrtest_MainActivity_getJniStringBytes(JNIEnv *env, jobject instance) {

    int byteCount = tts_result.length();
    jbyte * pNativeMessage = const_cast<jbyte *>(reinterpret_cast<const jbyte*>(tts_result.c_str()));
    jbyteArray bytes = env->NewByteArray(byteCount);
    env->SetByteArrayRegion(bytes, 0, byteCount, pNativeMessage);

    return bytes;

}
