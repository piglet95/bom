package com.example.Bom;

import android.annotation.TargetApi;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Build;
import android.os.CountDownTimer;
import android.os.Handler;
import android.os.Vibrator;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.support.annotation.NonNull;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.PriorityQueue;

import static android.speech.tts.TextToSpeech.ERROR;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "OPENCV::";
    private CameraBridgeViewBase mOpenCvCameraView;

    private Mat matInput;
    private Mat matGray;
    private Mat image_matches;

    private List<Mat> matArrayList;
    private List<Mat> tempMatArrayList;
    private List<InputStream> inputStreamArrayList;
    private List<Bitmap> bitmapArrayList;

    private long[] tempAddrObj;
    private String imgFilesName[] = {"1000won_back.jpeg", "1000won_front.jpeg", "5000won_back.jpeg", "5000won_front.jpeg", "10000won_back.jpeg", "10000won_front.jpeg", "50000won_back.jpeg", "50000won_front.jpeg"};

    private Button objects_button;
    private Button banknotes_button;
    private Button detection_button;
    private String pressedBtn = "";

    private Button tutorial_button;

    private TextToSpeech tts;
    private String banknotes_result_tts;

    private Vibrator start_vibrator;

    // YOLO
    private static final String YOLO_MODEL_FILE = "file:///android_asset/yolo-voc-45000.pb";
    private static final int YOLO_INPUT_SIZE = 416;
    private static final String YOLO_INPUT_NAME = "input";
    private static final String YOLO_OUTPUT_NAMES = "output";
    private static final int YOLO_BLOCK_SIZE = 32;
    private static final String[] LABELS = {
            "Neoguri",
            "ShinRamyun",
            "Kkongchi",
            "Godeungeo",
            "Kkaennip",
            "Myeongi"
    };
    private static final String[] KOREANLABELS = {
            "너구리",
            "신라면",
            "꽁치",
            "고등어",
            "깻잎",
            "명이"
    };

    private static final int MAX_RESULTS = 5;
    private static final float MINIMUM_CONFIDENCE_YOLO = 0.6f;
    private Bitmap matBitmap = null;
    private Classifier detector;

    // Object Detection
    private LinkedHashSet<Integer> linkedHashSet = new LinkedHashSet<>();
    private Iterator<Integer> iter;
    private List<Integer> beforeArrayList = new ArrayList<Integer> ();
    private List<Integer> afterArrayList = new ArrayList<Integer>();
    private PriorityQueue<ObjectQueue> objectQueues = new PriorityQueue<>(MAX_RESULTS, new SortQueueViaPriority());
    private ObjectQueue objectQueue;
    private int objectQueuesSize;
    private int objectDetectedClass;
    private int beforeArrayListSize = 0;
    private int afterArrayListSize = 0;

    // Countdown Timer
    private static final int MILLISINFUTURE = 10 * 1000;
    private static final int COUNT_DOWN_INTERVAL = 1000;
    private int cnt = 0;
    private CountDownTimer countDownTimer;
    private Vibrator vibrator;
    private String key;
    private ArrayList<String> mResult;
    private String[] rs = new String[60];

    //STT
    private Intent intent;
    private SpeechRecognizer mRecognizer;

    private Handler handler;
    private String final_result = "";
    private HashMap<String, Integer> hashMap = new HashMap<>();

    //TTS
    private TextToSpeech textToSpeech;
    private String lastRecognizedClass = "";


    // JNI
    public native void surfWithFlann(long matAddrInput, long image_matches);
    public native void sendImages(long[] tempAddrObj);
    public native byte[] getJniStringBytes();

    static {
        System.loadLibrary("opencv_java3");
        System.loadLibrary("native-lib");

    }


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    mOpenCvCameraView.enableView();
                }
                break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.activity_main);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (!hasPermissions(PERMISSIONS)) {
                requestPermissions(PERMISSIONS, PERMISSIONS_REQUEST_CODE);
            }
        }

        sendImagesToJNI();

        sendImages(tempAddrObj);

        //STT
        intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, getPackageName());
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE,"ko-KR");
        intent.putExtra (RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 3000);
        intent.putExtra (RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 3000);

        matInput = new Mat();
        matGray = new Mat();
        image_matches = new Mat();

        mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.activity_surface_view);
        mOpenCvCameraView.enableFpsMeter();
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraIndex(0); // front-camera(1), back-camera(0)
        mOpenCvCameraView.setMaxFrameSize(1280, 720);
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);

        handler = new Handler();
        mRecognizer = (SpeechRecognizer) SpeechRecognizer.createSpeechRecognizer(MainActivity.this);

        vibrator = (Vibrator)getSystemService(Context.VIBRATOR_SERVICE);

        start_vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        if (start_vibrator.hasVibrator()) {
            start_vibrator.vibrate(1000); // vibrate for 1000 ms
        }

        for(int i = 0; i < LABELS.length; i++) {
            hashMap.put(LABELS[i], 0);
        }

        tts = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status != ERROR) {
                    tts.setLanguage(Locale.KOREAN);
                }
            }
        });
        tts.setSpeechRate(0.85f);

        textToSpeech = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status != ERROR) {
                    textToSpeech.setLanguage(Locale.KOREAN);
                }
            }
        });
        textToSpeech.setSpeechRate(0.95f);

        objects_button = (Button) findViewById(R.id.objects);
        banknotes_button = (Button) findViewById(R.id.banknotes);
        detection_button = (Button) findViewById(R.id.detection);
        tutorial_button = (Button)findViewById(R.id.tutorial_button);

        objects_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pressedBtn = "사물";

                objects_button.setBackgroundResource(R.drawable.button1_change);

                banknotes_button.setBackgroundResource(R.drawable.button2);
                detection_button.setBackgroundResource(R.drawable.button3);

            }
        });

        banknotes_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pressedBtn = "지폐";

                banknotes_button.setBackgroundResource(R.drawable.button2_change);

                objects_button.setBackgroundResource(R.drawable.button1);
                detection_button.setBackgroundResource(R.drawable.button3);
            }
        });

        detection_button.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                pressedBtn = "탐지";

                detection_button.setBackgroundResource(R.drawable.button3_change);

                objects_button.setBackgroundResource(R.drawable.button1);
                banknotes_button.setBackgroundResource(R.drawable.button2);

                final_result = "";
                rs[0] = "";

                textToSpeech.speak("찾을 물건을 말해주세요. ", TextToSpeech.QUEUE_ADD, null);
                Toast.makeText(getApplicationContext(),
                        "찾을 물건을 말해주세요.", Toast.LENGTH_SHORT).show();

                handler.postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            mRecognizer.setRecognitionListener(listener);
                            mRecognizer.startListening(intent);

                        } catch (SecurityException e) {
                            e.printStackTrace();
                        }
                    }
                }, 1500);

            }

        });

        // Countdown 10 Seconds
        countDownTimer = new CountDownTimer(MILLISINFUTURE, COUNT_DOWN_INTERVAL){
            public void onTick(long millisUntilFinished){
                cnt++;
            }

            public void onFinish(){
                textToSpeech.speak("시간이 초과되었어요. 다시 찾으시려면 버튼을 눌러주세요. ", TextToSpeech.QUEUE_ADD, null);
                Toast.makeText(getApplicationContext(),
                        "시간 초과", Toast.LENGTH_SHORT).show();
            }
        };

        tutorial_button.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                textToSpeech.speak("사용법을 알려드리겠습니다. 버튼 별 사용법을 알려드리겠습니다. 첫 번째 사물 버튼은 실시간으로 사물을 인식하여 음성으로 인식한 물건을 왼쪽부터 순서대로 알려줍니다. 두 번째 지폐 버튼은 실시간으로 지폐의 금액을 음성으로 알려줍니다. 세 번째 탐지 버튼은 '찾을 물건을 말해주세요' 라는 안내 이후 찾을 물건을 말하면 카메라로 주변을 인식하여 찾은 물건을 진동으로 알려줍니다.", TextToSpeech.QUEUE_ADD, null);
            }
        });

        detector =
                TensorFlowYoloDetector.create(
                        getAssets(),
                        YOLO_MODEL_FILE,
                        YOLO_INPUT_SIZE,
                        YOLO_INPUT_NAME,
                        YOLO_OUTPUT_NAMES,
                        YOLO_BLOCK_SIZE);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();

        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }

        if(mRecognizer != null) {
            mRecognizer.destroy();
        }

        super.onDestroy();

        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {
        matInput.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        matInput = inputFrame.rgba();

        if(pressedBtn.equals("사물")) {
            matBitmap = Bitmap.createBitmap(matInput.cols(), matInput.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(matInput, matBitmap);
            matBitmap = Bitmap.createScaledBitmap(matBitmap, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, true);

            objectFunction();

            return matInput;
        }

        if(pressedBtn.equals("지폐")) {
            Imgproc.cvtColor(matInput, matGray, Imgproc.COLOR_RGB2GRAY);

            new SURFAsyncTask().execute(matGray, image_matches);

            return matInput;
        }

        if(pressedBtn.equals("탐지")) {
            matBitmap = Bitmap.createBitmap(matInput.cols(), matInput.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(matInput, matBitmap);
            matBitmap = Bitmap.createScaledBitmap(matBitmap, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, true);

            detectionFunction();

        }


        return matInput;
    }

    // objectQueue에 contain 되어 있고 hashmap의 개수가 1개 이상인 경우

    public void objectFunction() {


        final List<Classifier.Recognition> recognitions = detector.recognizeImage(matBitmap);
        float minimumConfidence = MINIMUM_CONFIDENCE_YOLO;
        for (final Classifier.Recognition result : recognitions) {
            if (result.getConfidence() >= minimumConfidence) {
                objectQueues.add(new ObjectQueue(result.getTitle(), result.getXpos(), result.getDetectedClass()));
            }
        }


        objectQueuesSize = objectQueues.size();

        for(int i = 0; i < objectQueuesSize; i++) {
            objectQueue = objectQueues.poll();
            objectDetectedClass = objectQueue.detectedClass;
            /*화면에 신라면, 너구리만 있는 경운데 [신라면, 너구리, 신라면] 으로 인식되는 경우를 방지하기 위해 LinkedHashSet 사용
            LinkedHashSet은 순서를 지켜주는 Set임.*/
            linkedHashSet.add(objectDetectedClass);
        }


        tts.setSpeechRate(0.85f);

        /*iterator()를 사용하여 LinkedHashSet에 있는 값을 beforeArrayList에 추가해줌.*/
        iter = linkedHashSet.iterator();
        while(iter.hasNext()) {
            int objectNum = iter.next();
            beforeArrayList.add(objectNum);
        }

        linkedHashSet.clear();

        /*beforeArrayList는 새롭게 인식되는 사물들이 있는 ArrayList.
        afterArrayList는 이전에 인식되는 사물들이 있는 ArrayList.*/


        if(!beforeArrayList.isEmpty()) {

            if(afterArrayList.isEmpty()) {

                /*beforeArrayList에 있는 값들을 읽어줌.*/
                beforeArrayListSize = beforeArrayList.size();
                for (int i = 0; i < beforeArrayListSize; i++) {
                    tts.speak(KOREANLABELS[beforeArrayList.get(i)], TextToSpeech.QUEUE_ADD, null, null);
                }

                /*afterArrayList에 beforeArrayList의 값을 복사해줌.
                지금 beforeArrayList에 들어있는 값은 이제 새롭게 인식되는 사물들이 있으면 이전에 인식된 사물인 것.*/
                afterArrayList.addAll(beforeArrayList); /*깊은 복사*/

                /*새롭게 인식되는 사물들을 넣어야하기 때문에 clear 시켜줌.*/
                beforeArrayList.clear();

            } else {
                beforeArrayListSize = beforeArrayList.size();
                afterArrayListSize = afterArrayList.size();

                /*beforeArrayList와 afterArrayList의 사이즈를 비교함.*/
                if(beforeArrayListSize != afterArrayListSize) {

                    /*beforeArrayList에 있는 값을 그대로 읽어줌.*/
                    for(int i = 0; i < beforeArrayListSize; i++) {
                        tts.speak(KOREANLABELS[beforeArrayList.get(i)], TextToSpeech.QUEUE_ADD, null, null);

                    }

                    /*앞에 과정을 거쳐줌.*/
                    afterArrayList.clear();
                    afterArrayList.addAll(beforeArrayList);

                    beforeArrayList.clear();

                } else {

                    for(int i = 0; i < beforeArrayListSize; i++) {

                        /*ArrayList에 있는 i번째 값들을 비교.*/
                        if(beforeArrayList.get(i) == afterArrayList.get(i)) {

                        }
                        /*값이 다르면 바로 beforeArrayList에 값을 읽어주는 함수로 이동함.*/
                        else {
                            readBeforeArrayList();
                            break;
                        }
                    }

                    beforeArrayList.clear();
                }
            }
        }

    }



    public void readBeforeArrayList() {
        // beforeArrayList에 있는 값을 읽어줌.
        for(int i = 0; i < beforeArrayListSize; i++) {
            tts.speak(KOREANLABELS[beforeArrayList.get(i)], TextToSpeech.QUEUE_ADD, null, null);
        }

        // 앞에와 같은 과정을 거침.
        afterArrayList.clear();
        afterArrayList.addAll(beforeArrayList);

        beforeArrayList.clear();
    }

    public void detectionFunction() {

        final List<Classifier.Recognition> recognitions = detector.recognizeImage(matBitmap);
        float minimumConfidence = MINIMUM_CONFIDENCE_YOLO;
        for (final Classifier.Recognition result : recognitions) {
            if (result.getConfidence() >= minimumConfidence && final_result != result.getTitle() && rs[0] == result.getTitle()) {
                final_result = result.getTitle();
                speakResult(result);

            }

        }

    }

    protected void speakResult(Classifier.Recognition result) {
        if (!result.equals("")) {
            lastRecognizedClass = result.getTitle();

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                if(lastRecognizedClass.equals(rs[0]) && (rs[0] != null)){ // 인식한 클래스가 sst로 전달된 사물과 같으면

                    countDownTimer.cancel(); // 타이머 종료
                    vibrator.vibrate(3000);} // 3초 진동울림
            }
            else {
                tts.speak(lastRecognizedClass, TextToSpeech.QUEUE_FLUSH, null, null);
            }
        }
    }

    class SortQueueViaPriority implements Comparator<ObjectQueue> {
        @Override
        public int compare(ObjectQueue oq1, ObjectQueue oq2) {
            return Float.compare(oq1.getxPos(), oq2.getxPos());
        }
    }

    class ObjectQueue {
        private final String name;
        private final float xPos;
        private final int detectedClass;

        ObjectQueue(String name, float xPos, int detectedClass) {
            this.name = name;
            this.xPos = xPos;
            this.detectedClass = detectedClass;
        }

        public float getxPos() {
            return xPos;
        }

        @Override
        public String toString() {
            return "ObjectQueue{" + "name='" + name + '\'' + ", priority=" + xPos + ", detectedClass=" + detectedClass + '}';
        }
    }


    private class SURFAsyncTask extends AsyncTask<Mat, Void, Mat> {

        @Override
        protected Mat doInBackground(Mat... mats) {

            surfWithFlann(mats[0].getNativeObjAddr(), mats[1].getNativeObjAddr());
            Imgproc.resize(mats[1], mats[1], mats[0].size());

            banknotes_result_tts = new String(getJniStringBytes(), Charset.forName("UTF-8"));

            tts.speak(banknotes_result_tts, TextToSpeech.QUEUE_FLUSH, null);

            return mats[1];
        }

    }

    public void sendImagesToJNI() {
        AssetManager assetManager = getAssets();
        matArrayList = new ArrayList<>();
        inputStreamArrayList = new ArrayList<>();
        tempMatArrayList = new ArrayList<>();
        bitmapArrayList = new ArrayList<>();

        int imgFilesNameSize = imgFilesName.length;

        for(int i = 0; i < imgFilesNameSize; i++) {
            tempMatArrayList.add(new Mat());
        }

        try {

            for(int i = 0; i < imgFilesNameSize; i++) {
                inputStreamArrayList.add(assetManager.open(imgFilesName[i]));
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        for(int i = 0; i < imgFilesNameSize; i++) {
            bitmapArrayList.add(BitmapFactory.decodeStream(inputStreamArrayList.get(i)));
            Utils.bitmapToMat(bitmapArrayList.get(i), tempMatArrayList.get(i));
            Imgproc.cvtColor(tempMatArrayList.get(i), tempMatArrayList.get(i), Imgproc.COLOR_BGR2GRAY);
            matArrayList.add(tempMatArrayList.get(i));

        }


        int elems = matArrayList.size();
        tempAddrObj = new long[elems];

        for (int i = 0; i < elems; i++) {
            Mat tempAddrMat = matArrayList.get(i);
            tempAddrObj[i] = tempAddrMat.getNativeObjAddr();
        }

    }

    public RecognitionListener listener = new RecognitionListener(){
        @Override
        public void onBeginningOfSpeech() {
        }

        @Override
        public void onReadyForSpeech(Bundle params) {
            countDownTimer.start();
        }

        @Override
        public void onRmsChanged(float v) {
        }

        @Override
        public void onBufferReceived(byte[] bytes) {
        }

        @Override
        public void onEndOfSpeech() {
        }

        @Override
        public void onError(int i) {
        }


        @Override
        public void onResults(Bundle results) {
            key = SpeechRecognizer.RESULTS_RECOGNITION;
            mResult = results.getStringArrayList(key);

            rs = new String[mResult.size()];
            mResult.toArray(rs);
            Toast.makeText(MainActivity.this, rs[0],Toast.LENGTH_SHORT).show();

            //한글을 영어로 바꾸기
            if(rs[0].equals("너구리"))
                rs[0] = "Neoguri";

            else if(rs[0].equals("신라면"))
                rs[0] = "ShinRamyun";

            else if(rs[0].equals("꽁치"))
                rs[0] = "Kkongchi";

            else if(rs[0].equals("고등어"))
                rs[0] = "Godeungeo";

            else if(rs[0].equals("깻잎"))
                rs[0] = "Kkaennip";

            else if(rs[0].equals("명이"))
                rs[0] = "Myeongi";

        }

        @Override
        public void onPartialResults(Bundle bundle) {
        }

        @Override
        public void onEvent(int i, Bundle bundle) {
        }
    };

    static final int PERMISSIONS_REQUEST_CODE = 1000;
    String[] PERMISSIONS = {"android.permission.CAMERA"};


    private boolean hasPermissions(String[] permissions) {
        int result;

        for (String perms : permissions){
            result = ContextCompat.checkSelfPermission(this, perms);
            if (result == PackageManager.PERMISSION_DENIED){
                return false;
            }
        }

        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        switch(requestCode){

            case PERMISSIONS_REQUEST_CODE:
                if (grantResults.length > 0) {
                    boolean cameraPermissionAccepted = grantResults[0]
                            == PackageManager.PERMISSION_GRANTED;

                    if (!cameraPermissionAccepted)
                        showDialogForPermission("앱을 실행하려면 퍼미션을 허가하셔야합니다.");
                }
                break;
        }
    }

    @TargetApi(Build.VERSION_CODES.M)
    private void showDialogForPermission(String msg) {

        AlertDialog.Builder builder = new AlertDialog.Builder( MainActivity.this);
        builder.setTitle("알림");
        builder.setMessage(msg);
        builder.setCancelable(false);
        builder.setPositiveButton("예", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int id){
                requestPermissions(PERMISSIONS, PERMISSIONS_REQUEST_CODE);
            }
        });
        builder.setNegativeButton("아니오", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface arg0, int arg1) {
                finish();
            }
        });
        builder.create().show();
    }

}