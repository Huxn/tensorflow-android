package demo.unionpay.com.tensorflowdemo;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;


public class MainActivity extends AppCompatActivity {
    // Used to load the 'native-lib' library on application startup.
//    static {
//        System.loadLibrary("native-lib");//可以去掉
//    }

    private static final String TAG = "MainActivity";
    private static final String MODEL_FILE = "file:///android_asset/mnist.pb"; //模型存放路径
    TextView txt;
//    TextView tv;
    ImageView imageView;
    Bitmap bitmap;
    PredictionTF preTF;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Example of a call to a native method
//        tv = (TextView) findViewById(R.id.sample_text);
        txt=(TextView)findViewById(R.id.txt_id);
        imageView =(ImageView)findViewById(R.id.imageView1);
        bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.image);
        imageView.setImageBitmap(bitmap);
        preTF =new PredictionTF(getAssets(),MODEL_FILE);//输入模型存放路径，并加载TensoFlow模型
    }

    public void click01(View v){
        String res="预测结果为：\n";
        HashMap<Integer, Float> result= preTF.getPredict(bitmap);
        result = sort(result);
        int i = 0;
        for (int key: result.keySet()){
            i++;
            if (i > 3)
                break;
            res += ("[" + String.valueOf(key) + "," + String.valueOf(result.get(key)) + "]\n");
        }

        txt.setText(res.substring(0,res.length()-1));
//        tv.setText(stringFromJNI());
    }

    private HashMap<Integer, Float> sort(HashMap<Integer, Float> result) {
        HashMap<Integer, Float> sortedResult = new LinkedHashMap<>();
        List<Entry<Integer, Float>> list = new LinkedList<>(result.entrySet());
        Collections.sort(list, new Comparator<Entry<Integer, Float>>() {
            @Override
            public int compare(Entry<Integer, Float> e1, Entry<Integer, Float> e2) {
                if (e1.getValue() > e2.getValue())
                    return -1;
                else if(e1.getValue() < e2.getValue())
                    return 1;
                else
                    return 0;
            }
        });
        for(Entry<Integer, Float> entry: list)
            sortedResult.put(entry.getKey(), entry.getValue());
        return sortedResult;
    }
//    /**
//     * A native method that is implemented by the 'native-lib' native library,
//     * which is packaged with this application.
//     */
//    public native String stringFromJNI();//可以去掉

}
