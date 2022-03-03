```python
import tensorflow as tf
import numpy as np
```

### Fashion MNIST 데이터셋 불러오기 
* load_data()함수를 호출하면 numpy 튜플 형태로 반환


```python
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
    32768/29515 [=================================] - 0s 1us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
    26427392/26421880 [==============================] - 14s 1us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
    8192/5148 [===============================================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
    4423680/4422102 [==============================] - 2s 0us/step
    

### 첫번째 변수 살펴보기


```python
print(train_images[0])
print(train_labels[0])
```

    [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   1   0   0  13  73   0
        0   1   4   0   0   0   0   1   1   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   3   0  36 136 127  62
       54   0   0   0   1   3   4   0   0   3]
     [  0   0   0   0   0   0   0   0   0   0   0   0   6   0 102 204 176 134
      144 123  23   0   0   0   0  12  10   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0 155 236 207 178
      107 156 161 109  64  23  77 130  72  15]
     [  0   0   0   0   0   0   0   0   0   0   0   1   0  69 207 223 218 216
      216 163 127 121 122 146 141  88 172  66]
     [  0   0   0   0   0   0   0   0   0   1   1   1   0 200 232 232 233 229
      223 223 215 213 164 127 123 196 229   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0 183 225 216 223 228
      235 227 224 222 224 221 223 245 173   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0 193 228 218 213 198
      180 212 210 211 213 223 220 243 202   0]
     [  0   0   0   0   0   0   0   0   0   1   3   0  12 219 220 212 218 192
      169 227 208 218 224 212 226 197 209  52]
     [  0   0   0   0   0   0   0   0   0   0   6   0  99 244 222 220 218 203
      198 221 215 213 222 220 245 119 167  56]
     [  0   0   0   0   0   0   0   0   0   4   0   0  55 236 228 230 228 240
      232 213 218 223 234 217 217 209  92   0]
     [  0   0   1   4   6   7   2   0   0   0   0   0 237 226 217 223 222 219
      222 221 216 223 229 215 218 255  77   0]
     [  0   3   0   0   0   0   0   0   0  62 145 204 228 207 213 221 218 208
      211 218 224 223 219 215 224 244 159   0]
     [  0   0   0   0  18  44  82 107 189 228 220 222 217 226 200 205 211 230
      224 234 176 188 250 248 233 238 215   0]
     [  0  57 187 208 224 221 224 208 204 214 208 209 200 159 245 193 206 223
      255 255 221 234 221 211 220 232 246   0]
     [  3 202 228 224 221 211 211 214 205 205 205 220 240  80 150 255 229 221
      188 154 191 210 204 209 222 228 225   0]
     [ 98 233 198 210 222 229 229 234 249 220 194 215 217 241  65  73 106 117
      168 219 221 215 217 223 223 224 229  29]
     [ 75 204 212 204 193 205 211 225 216 185 197 206 198 213 240 195 227 245
      239 223 218 212 209 222 220 221 230  67]
     [ 48 203 183 194 213 197 185 190 194 192 202 214 219 221 220 236 225 216
      199 206 186 181 177 172 181 205 206 115]
     [  0 122 219 193 179 171 183 196 204 210 213 207 211 210 200 196 194 191
      195 191 198 192 176 156 167 177 210  92]
     [  0   0  74 189 212 191 175 172 175 181 185 188 189 188 193 198 204 209
      210 210 211 188 188 194 192 216 170   0]
     [  2   0   0   0  66 200 222 237 239 242 246 243 244 221 220 193 191 179
      182 182 181 176 166 168  99  58   0   0]
     [  0   0   0   0   0   0   0  40  61  44  72  41  35   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0]]
    9
    


```python
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
```

    (60000, 28, 28)
    (60000,)
    (10000, 28, 28)
    (10000,)
    

train_images는 0에서 255사이의 값을 갖는 28*28크기 형태 6만개 numpy 배열 구성  
test_images는 마찬가지의 형태로 만개의 이미지 배열 구성  
trian_labels는 0에서 9까지의 정수를 갖고 있는 배열 구성
  
train_labels는 아래와 같은 분류로 됨  
0 : T-shirt/top  
1 : Trouser  
2 : Pullover  
3 : Dress  
4 : Coat  
5 : Sandal  
6 : Shirt  
7 : Sneaker  
8 : Bag  
9 : Ankel boot  

### 데이터 셋 전처리

0에서 255사이의 값 갖는 데이터들을 0에서 1사이의 값 갖도록 설정


```python
train_images, test_images = train_images / 255.0, test_images / 255.0
```

### 모델 구성


```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), #입력데이터 1차원으로 변환
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### 모델 컴파일하기
손실함수, 지표 설정
* 손실함수(loss) : 모델의 오차 측정시 사용
* 옵티마이저(optimizer) : 데이터와 손실함수 바탕으로 모델 업데이트 하는 방식 
* 지표(metrics) : 훈련과 테스트 단계 평가위한 사용  


```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) #acccuracy는 이미지 분류 모델 평가
```

### 모델 훈련하기 
epochs 통해 전체 이미지 데이터 몇번 학습할지 설정

![%EC%BA%A1%EC%B2%98.PNG](attachment:%EC%BA%A1%EC%B2%98.PNG)


```python
model.fit(train_images,train_labels,epochs=5)
```

    Epoch 1/5
    1875/1875 [==============================] - 17s 8ms/step - loss: 0.4771 - accuracy: 0.8294
    Epoch 2/5
    1875/1875 [==============================] - 19s 10ms/step - loss: 0.3592 - accuracy: 0.8691
    Epoch 3/5
    1875/1875 [==============================] - 22s 12ms/step - loss: 0.3219 - accuracy: 0.8825
    Epoch 4/5
    1875/1875 [==============================] - 16s 9ms/step - loss: 0.2991 - accuracy: 0.8899
    Epoch 5/5
    1875/1875 [==============================] - 15s 8ms/step - loss: 0.2793 - accuracy: 0.8961
    




    <tensorflow.python.keras.callbacks.History at 0x18a7db67940>



### 정확도 평가하기


```python
loss, accuracy = model.evaluate(test_images, test_labels)
print(loss, accuracy)
```

    313/313 [==============================] - 3s 4ms/step - loss: 0.3290 - accuracy: 0.8799
    0.3290295898914337 0.8798999786376953
    

* 5회 epochs 학습 통해 만개의 test_images 87퍼센트의 정확도로 분류한 것을 확인할 수 있음 

### 각 이미지의 클래스 예측해보기


```python
predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
```

    [6.6468908e-07 2.3154195e-08 2.5811744e-07 1.9266940e-06 2.4186619e-07
     3.6460210e-02 1.3099501e-06 1.9631973e-02 1.0628785e-06 9.4390243e-01]
    9
    

위 결과는 10개의 배열이 어떤 값을 갖는지 보여주며  
입력 이미지 데이터가 열개의 숫자 중 어떤 숫자일 확률을 의미  
현재 보면 9.439로 9번째 숫자의 값이 가장 큰 것을 볼 수 있고,  
np.argmax()를 통해 가장 높은 값을 갖는 인덱스 역시 9라고 보여주고 있음 
  
-->따라서 0번째의 학습 이미지는 ankle boot라고 예측됨

### 모델 다르게 구성해보기


```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),  
    tf.keras.layers.Dense(10, activation='softmax')
])
```


```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) #acccuracy는 이미지 분류 모델 평가
```


```python
model.fit(train_images,train_labels,epochs=5)
```

    Epoch 1/5
    1875/1875 [==============================] - 8s 3ms/step - loss: 0.4956 - accuracy: 0.8256
    Epoch 2/5
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.3747 - accuracy: 0.8648
    Epoch 3/5
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.3355 - accuracy: 0.8768
    Epoch 4/5
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.3115 - accuracy: 0.8856
    Epoch 5/5
    1875/1875 [==============================] - 6s 3ms/step - loss: 0.2937 - accuracy: 0.8906
    




    <tensorflow.python.keras.callbacks.History at 0x18a7e3743d0>




```python
loss, accuracy = model.evaluate(test_images, test_labels)
print(loss, accuracy)
```

    313/313 [==============================] - 1s 4ms/step - loss: 0.3519 - accuracy: 0.8743
    0.35188016295433044 0.8743000030517578
    

아까와 비교해보면 손실값이 0.03  정확도가 0.005정도 떨어진 것을 확인할 수 있다

노드의 개수가 증가하면 손실값이 감소하고 테스트 정확도는 증가하는 경향을 보이고 있지만  
노드가 증가할수록 걸리는 시간 역시 증가할 수 있다.  

