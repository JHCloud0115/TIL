https://www.kaggle.com/code/sergiogarridomerino/clasificaci-n-de-comida

## 라이브러리 불러오기


```python
import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import glob

from sklearn.model_selection import train_test_split

import tensorflow as tf
```

## 데이터 불러오기


```python
img_dir = Path('D:\images')
img_dir
```




    WindowsPath('D:/images')




```python
images_paths = list(img_dir.glob(r'**/*.jpg'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], images_paths))

images_paths = pd.Series(images_paths, name = 'Image').astype(str)
labels = pd.Series(labels, name='Label')

images = pd.concat([images_paths, labels], axis=1)
```


```python
images.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Image</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>D:\images\apple_pie\1005649.jpg</td>
      <td>apple_pie</td>
    </tr>
    <tr>
      <th>1</th>
      <td>D:\images\apple_pie\1011328.jpg</td>
      <td>apple_pie</td>
    </tr>
    <tr>
      <th>2</th>
      <td>D:\images\apple_pie\101251.jpg</td>
      <td>apple_pie</td>
    </tr>
  </tbody>
</table>
</div>



### Data Set 설정


```python
category_samples = []
for category in images['Label'].unique():
    category_slice = images.query('Label == @category')
    category_samples.append(category_slice.sample(300, random_state = 1))

#그룹화하여 범주별 이미지 수 확인 
images_samples = pd.concat(category_samples, axis = 0)
images_samples['Label'].value_counts()
```




    apple_pie        300
    miso_soup        300
    peking_duck      300
    panna_cotta      300
    pancakes         300
                    ... 
    donuts           300
    deviled_eggs     300
    cup_cakes        300
    croque_madame    300
    waffles          300
    Name: Label, Length: 101, dtype: int64




```python
train_data, test_data = train_test_split(images_samples, train_size = 0.7, shuffle = True, random_state = 1)
```

### ImageDataGenerator
학습 데이터의 다양성을 늘리게 되면 오버피팅의 문제점을 해결할 수 있음   
따라서 오버피팅의 문제점을 해결하기 위해서 나온 방법 중 하나인  
**Data Augmentation, 데이터 증강 기법**    
하나의 원본 이미지를 다양한 비전(수직 반전, 회전 등)으로 만들어 학습시키는 것


```python
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split = 0.2
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
)
```


```python
train_images = train_generator.flow_from_dataframe(
    dataframe = train_data,#불러올 데이터프레임
    x_col = 'Image', #파일위치 열이름
    y_col = 'Label', #클래스 열이름
    target_size = (224, 224), #이미지 사이즈
    color_mode = 'rgb', #이미지 채널 수 
    class_mode = 'categorical', #y값 변화방법 
    batch_size = 32, 
    shuffle = True,
    seed = 42,
    subset = 'training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe = train_data,
    x_col = 'Image',
    y_col = 'Label',
    target_size = (224, 224),
    color_mode = 'rgb',
    class_mode = 'categorical',
    batch_size = 32,
    shuffle = True,
    seed = 42,
    subset = 'training'
)

test_images = test_generator.flow_from_dataframe(
    dataframe = test_data,
    x_col = 'Image',
    y_col = 'Label',
    target_size = (224, 224),
    color_mode = 'rgb',
    class_mode = 'categorical',
    batch_size = 32,
    shuffle = False
)
```

    Found 16968 validated image filenames belonging to 101 classes.
    Found 16968 validated image filenames belonging to 101 classes.
    Found 9090 validated image filenames belonging to 101 classes.
    

## 모델 생성


```python
pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape = (224, 224, 3), # 입력 이미지 해상도 기본이 224 224 3 
    include_top = False, #완전연결계층 사용여부 
    weights = 'imagenet', #imagenet에 대한 사전학습된 가중치 파일의 경로 
    pooling = 'avg', #문자열인 경우 사용 
)

# 
pretrained_model.trainable = False
```


```python
inputs = pretrained_model.input

#layer 설정 
x = tf.keras.layers.Dense(128, activation = 'relu')(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation = 'relu')(x)

#101개의 라벨 클래스 
output_layer = tf.keras.layers.Dense(101, activation = 'softmax')(x)


model = tf.keras.Model(inputs, output_layer)

print(model.summary())
```

    Model: "model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_1 (InputLayer)           [(None, 224, 224, 3  0           []                               
                                    )]                                                                
                                                                                                      
     Conv1 (Conv2D)                 (None, 112, 112, 32  864         ['input_1[0][0]']                
                                    )                                                                 
                                                                                                      
     bn_Conv1 (BatchNormalization)  (None, 112, 112, 32  128         ['Conv1[0][0]']                  
                                    )                                                                 
                                                                                                      
     Conv1_relu (ReLU)              (None, 112, 112, 32  0           ['bn_Conv1[0][0]']               
                                    )                                                                 
                                                                                                      
     expanded_conv_depthwise (Depth  (None, 112, 112, 32  288        ['Conv1_relu[0][0]']             
     wiseConv2D)                    )                                                                 
                                                                                                      
     expanded_conv_depthwise_BN (Ba  (None, 112, 112, 32  128        ['expanded_conv_depthwise[0][0]']
     tchNormalization)              )                                                                 
                                                                                                      
     expanded_conv_depthwise_relu (  (None, 112, 112, 32  0          ['expanded_conv_depthwise_BN[0][0
     ReLU)                          )                                ]']                              
                                                                                                      
     expanded_conv_project (Conv2D)  (None, 112, 112, 16  512        ['expanded_conv_depthwise_relu[0]
                                    )                                [0]']                            
                                                                                                      
     expanded_conv_project_BN (Batc  (None, 112, 112, 16  64         ['expanded_conv_project[0][0]']  
     hNormalization)                )                                                                 
                                                                                                      
     block_1_expand (Conv2D)        (None, 112, 112, 96  1536        ['expanded_conv_project_BN[0][0]'
                                    )                                ]                                
                                                                                                      
     block_1_expand_BN (BatchNormal  (None, 112, 112, 96  384        ['block_1_expand[0][0]']         
     ization)                       )                                                                 
                                                                                                      
     block_1_expand_relu (ReLU)     (None, 112, 112, 96  0           ['block_1_expand_BN[0][0]']      
                                    )                                                                 
                                                                                                      
     block_1_pad (ZeroPadding2D)    (None, 113, 113, 96  0           ['block_1_expand_relu[0][0]']    
                                    )                                                                 
                                                                                                      
     block_1_depthwise (DepthwiseCo  (None, 56, 56, 96)  864         ['block_1_pad[0][0]']            
     nv2D)                                                                                            
                                                                                                      
     block_1_depthwise_BN (BatchNor  (None, 56, 56, 96)  384         ['block_1_depthwise[0][0]']      
     malization)                                                                                      
                                                                                                      
     block_1_depthwise_relu (ReLU)  (None, 56, 56, 96)   0           ['block_1_depthwise_BN[0][0]']   
                                                                                                      
     block_1_project (Conv2D)       (None, 56, 56, 24)   2304        ['block_1_depthwise_relu[0][0]'] 
                                                                                                      
     block_1_project_BN (BatchNorma  (None, 56, 56, 24)  96          ['block_1_project[0][0]']        
     lization)                                                                                        
                                                                                                      
     block_2_expand (Conv2D)        (None, 56, 56, 144)  3456        ['block_1_project_BN[0][0]']     
                                                                                                      
     block_2_expand_BN (BatchNormal  (None, 56, 56, 144)  576        ['block_2_expand[0][0]']         
     ization)                                                                                         
                                                                                                      
     block_2_expand_relu (ReLU)     (None, 56, 56, 144)  0           ['block_2_expand_BN[0][0]']      
                                                                                                      
     block_2_depthwise (DepthwiseCo  (None, 56, 56, 144)  1296       ['block_2_expand_relu[0][0]']    
     nv2D)                                                                                            
                                                                                                      
     block_2_depthwise_BN (BatchNor  (None, 56, 56, 144)  576        ['block_2_depthwise[0][0]']      
     malization)                                                                                      
                                                                                                      
     block_2_depthwise_relu (ReLU)  (None, 56, 56, 144)  0           ['block_2_depthwise_BN[0][0]']   
                                                                                                      
     block_2_project (Conv2D)       (None, 56, 56, 24)   3456        ['block_2_depthwise_relu[0][0]'] 
                                                                                                      
     block_2_project_BN (BatchNorma  (None, 56, 56, 24)  96          ['block_2_project[0][0]']        
     lization)                                                                                        
                                                                                                      
     block_2_add (Add)              (None, 56, 56, 24)   0           ['block_1_project_BN[0][0]',     
                                                                      'block_2_project_BN[0][0]']     
                                                                                                      
     block_3_expand (Conv2D)        (None, 56, 56, 144)  3456        ['block_2_add[0][0]']            
                                                                                                      
     block_3_expand_BN (BatchNormal  (None, 56, 56, 144)  576        ['block_3_expand[0][0]']         
     ization)                                                                                         
                                                                                                      
     block_3_expand_relu (ReLU)     (None, 56, 56, 144)  0           ['block_3_expand_BN[0][0]']      
                                                                                                      
     block_3_pad (ZeroPadding2D)    (None, 57, 57, 144)  0           ['block_3_expand_relu[0][0]']    
                                                                                                      
     block_3_depthwise (DepthwiseCo  (None, 28, 28, 144)  1296       ['block_3_pad[0][0]']            
     nv2D)                                                                                            
                                                                                                      
     block_3_depthwise_BN (BatchNor  (None, 28, 28, 144)  576        ['block_3_depthwise[0][0]']      
     malization)                                                                                      
                                                                                                      
     block_3_depthwise_relu (ReLU)  (None, 28, 28, 144)  0           ['block_3_depthwise_BN[0][0]']   
                                                                                                      
     block_3_project (Conv2D)       (None, 28, 28, 32)   4608        ['block_3_depthwise_relu[0][0]'] 
                                                                                                      
     block_3_project_BN (BatchNorma  (None, 28, 28, 32)  128         ['block_3_project[0][0]']        
     lization)                                                                                        
                                                                                                      
     block_4_expand (Conv2D)        (None, 28, 28, 192)  6144        ['block_3_project_BN[0][0]']     
                                                                                                      
     block_4_expand_BN (BatchNormal  (None, 28, 28, 192)  768        ['block_4_expand[0][0]']         
     ization)                                                                                         
                                                                                                      
     block_4_expand_relu (ReLU)     (None, 28, 28, 192)  0           ['block_4_expand_BN[0][0]']      
                                                                                                      
     block_4_depthwise (DepthwiseCo  (None, 28, 28, 192)  1728       ['block_4_expand_relu[0][0]']    
     nv2D)                                                                                            
                                                                                                      
     block_4_depthwise_BN (BatchNor  (None, 28, 28, 192)  768        ['block_4_depthwise[0][0]']      
     malization)                                                                                      
                                                                                                      
     block_4_depthwise_relu (ReLU)  (None, 28, 28, 192)  0           ['block_4_depthwise_BN[0][0]']   
                                                                                                      
     block_4_project (Conv2D)       (None, 28, 28, 32)   6144        ['block_4_depthwise_relu[0][0]'] 
                                                                                                      
     block_4_project_BN (BatchNorma  (None, 28, 28, 32)  128         ['block_4_project[0][0]']        
     lization)                                                                                        
                                                                                                      
     block_4_add (Add)              (None, 28, 28, 32)   0           ['block_3_project_BN[0][0]',     
                                                                      'block_4_project_BN[0][0]']     
                                                                                                      
     block_5_expand (Conv2D)        (None, 28, 28, 192)  6144        ['block_4_add[0][0]']            
                                                                                                      
     block_5_expand_BN (BatchNormal  (None, 28, 28, 192)  768        ['block_5_expand[0][0]']         
     ization)                                                                                         
                                                                                                      
     block_5_expand_relu (ReLU)     (None, 28, 28, 192)  0           ['block_5_expand_BN[0][0]']      
                                                                                                      
     block_5_depthwise (DepthwiseCo  (None, 28, 28, 192)  1728       ['block_5_expand_relu[0][0]']    
     nv2D)                                                                                            
                                                                                                      
     block_5_depthwise_BN (BatchNor  (None, 28, 28, 192)  768        ['block_5_depthwise[0][0]']      
     malization)                                                                                      
                                                                                                      
     block_5_depthwise_relu (ReLU)  (None, 28, 28, 192)  0           ['block_5_depthwise_BN[0][0]']   
                                                                                                      
     block_5_project (Conv2D)       (None, 28, 28, 32)   6144        ['block_5_depthwise_relu[0][0]'] 
                                                                                                      
     block_5_project_BN (BatchNorma  (None, 28, 28, 32)  128         ['block_5_project[0][0]']        
     lization)                                                                                        
                                                                                                      
     block_5_add (Add)              (None, 28, 28, 32)   0           ['block_4_add[0][0]',            
                                                                      'block_5_project_BN[0][0]']     
                                                                                                      
     block_6_expand (Conv2D)        (None, 28, 28, 192)  6144        ['block_5_add[0][0]']            
                                                                                                      
     block_6_expand_BN (BatchNormal  (None, 28, 28, 192)  768        ['block_6_expand[0][0]']         
     ization)                                                                                         
                                                                                                      
     block_6_expand_relu (ReLU)     (None, 28, 28, 192)  0           ['block_6_expand_BN[0][0]']      
                                                                                                      
     block_6_pad (ZeroPadding2D)    (None, 29, 29, 192)  0           ['block_6_expand_relu[0][0]']    
                                                                                                      
     block_6_depthwise (DepthwiseCo  (None, 14, 14, 192)  1728       ['block_6_pad[0][0]']            
     nv2D)                                                                                            
                                                                                                      
     block_6_depthwise_BN (BatchNor  (None, 14, 14, 192)  768        ['block_6_depthwise[0][0]']      
     malization)                                                                                      
                                                                                                      
     block_6_depthwise_relu (ReLU)  (None, 14, 14, 192)  0           ['block_6_depthwise_BN[0][0]']   
                                                                                                      
     block_6_project (Conv2D)       (None, 14, 14, 64)   12288       ['block_6_depthwise_relu[0][0]'] 
                                                                                                      
     block_6_project_BN (BatchNorma  (None, 14, 14, 64)  256         ['block_6_project[0][0]']        
     lization)                                                                                        
                                                                                                      
     block_7_expand (Conv2D)        (None, 14, 14, 384)  24576       ['block_6_project_BN[0][0]']     
                                                                                                      
     block_7_expand_BN (BatchNormal  (None, 14, 14, 384)  1536       ['block_7_expand[0][0]']         
     ization)                                                                                         
                                                                                                      
     block_7_expand_relu (ReLU)     (None, 14, 14, 384)  0           ['block_7_expand_BN[0][0]']      
                                                                                                      
     block_7_depthwise (DepthwiseCo  (None, 14, 14, 384)  3456       ['block_7_expand_relu[0][0]']    
     nv2D)                                                                                            
                                                                                                      
     block_7_depthwise_BN (BatchNor  (None, 14, 14, 384)  1536       ['block_7_depthwise[0][0]']      
     malization)                                                                                      
                                                                                                      
     block_7_depthwise_relu (ReLU)  (None, 14, 14, 384)  0           ['block_7_depthwise_BN[0][0]']   
                                                                                                      
     block_7_project (Conv2D)       (None, 14, 14, 64)   24576       ['block_7_depthwise_relu[0][0]'] 
                                                                                                      
     block_7_project_BN (BatchNorma  (None, 14, 14, 64)  256         ['block_7_project[0][0]']        
     lization)                                                                                        
                                                                                                      
     block_7_add (Add)              (None, 14, 14, 64)   0           ['block_6_project_BN[0][0]',     
                                                                      'block_7_project_BN[0][0]']     
                                                                                                      
     block_8_expand (Conv2D)        (None, 14, 14, 384)  24576       ['block_7_add[0][0]']            
                                                                                                      
     block_8_expand_BN (BatchNormal  (None, 14, 14, 384)  1536       ['block_8_expand[0][0]']         
     ization)                                                                                         
                                                                                                      
     block_8_expand_relu (ReLU)     (None, 14, 14, 384)  0           ['block_8_expand_BN[0][0]']      
                                                                                                      
     block_8_depthwise (DepthwiseCo  (None, 14, 14, 384)  3456       ['block_8_expand_relu[0][0]']    
     nv2D)                                                                                            
                                                                                                      
     block_8_depthwise_BN (BatchNor  (None, 14, 14, 384)  1536       ['block_8_depthwise[0][0]']      
     malization)                                                                                      
                                                                                                      
     block_8_depthwise_relu (ReLU)  (None, 14, 14, 384)  0           ['block_8_depthwise_BN[0][0]']   
                                                                                                      
     block_8_project (Conv2D)       (None, 14, 14, 64)   24576       ['block_8_depthwise_relu[0][0]'] 
                                                                                                      
     block_8_project_BN (BatchNorma  (None, 14, 14, 64)  256         ['block_8_project[0][0]']        
     lization)                                                                                        
                                                                                                      
     block_8_add (Add)              (None, 14, 14, 64)   0           ['block_7_add[0][0]',            
                                                                      'block_8_project_BN[0][0]']     
                                                                                                      
     block_9_expand (Conv2D)        (None, 14, 14, 384)  24576       ['block_8_add[0][0]']            
                                                                                                      
     block_9_expand_BN (BatchNormal  (None, 14, 14, 384)  1536       ['block_9_expand[0][0]']         
     ization)                                                                                         
                                                                                                      
     block_9_expand_relu (ReLU)     (None, 14, 14, 384)  0           ['block_9_expand_BN[0][0]']      
                                                                                                      
     block_9_depthwise (DepthwiseCo  (None, 14, 14, 384)  3456       ['block_9_expand_relu[0][0]']    
     nv2D)                                                                                            
                                                                                                      
     block_9_depthwise_BN (BatchNor  (None, 14, 14, 384)  1536       ['block_9_depthwise[0][0]']      
     malization)                                                                                      
                                                                                                      
     block_9_depthwise_relu (ReLU)  (None, 14, 14, 384)  0           ['block_9_depthwise_BN[0][0]']   
                                                                                                      
     block_9_project (Conv2D)       (None, 14, 14, 64)   24576       ['block_9_depthwise_relu[0][0]'] 
                                                                                                      
     block_9_project_BN (BatchNorma  (None, 14, 14, 64)  256         ['block_9_project[0][0]']        
     lization)                                                                                        
                                                                                                      
     block_9_add (Add)              (None, 14, 14, 64)   0           ['block_8_add[0][0]',            
                                                                      'block_9_project_BN[0][0]']     
                                                                                                      
     block_10_expand (Conv2D)       (None, 14, 14, 384)  24576       ['block_9_add[0][0]']            
                                                                                                      
     block_10_expand_BN (BatchNorma  (None, 14, 14, 384)  1536       ['block_10_expand[0][0]']        
     lization)                                                                                        
                                                                                                      
     block_10_expand_relu (ReLU)    (None, 14, 14, 384)  0           ['block_10_expand_BN[0][0]']     
                                                                                                      
     block_10_depthwise (DepthwiseC  (None, 14, 14, 384)  3456       ['block_10_expand_relu[0][0]']   
     onv2D)                                                                                           
                                                                                                      
     block_10_depthwise_BN (BatchNo  (None, 14, 14, 384)  1536       ['block_10_depthwise[0][0]']     
     rmalization)                                                                                     
                                                                                                      
     block_10_depthwise_relu (ReLU)  (None, 14, 14, 384)  0          ['block_10_depthwise_BN[0][0]']  
                                                                                                      
     block_10_project (Conv2D)      (None, 14, 14, 96)   36864       ['block_10_depthwise_relu[0][0]']
                                                                                                      
     block_10_project_BN (BatchNorm  (None, 14, 14, 96)  384         ['block_10_project[0][0]']       
     alization)                                                                                       
                                                                                                      
     block_11_expand (Conv2D)       (None, 14, 14, 576)  55296       ['block_10_project_BN[0][0]']    
                                                                                                      
     block_11_expand_BN (BatchNorma  (None, 14, 14, 576)  2304       ['block_11_expand[0][0]']        
     lization)                                                                                        
                                                                                                      
     block_11_expand_relu (ReLU)    (None, 14, 14, 576)  0           ['block_11_expand_BN[0][0]']     
                                                                                                      
     block_11_depthwise (DepthwiseC  (None, 14, 14, 576)  5184       ['block_11_expand_relu[0][0]']   
     onv2D)                                                                                           
                                                                                                      
     block_11_depthwise_BN (BatchNo  (None, 14, 14, 576)  2304       ['block_11_depthwise[0][0]']     
     rmalization)                                                                                     
                                                                                                      
     block_11_depthwise_relu (ReLU)  (None, 14, 14, 576)  0          ['block_11_depthwise_BN[0][0]']  
                                                                                                      
     block_11_project (Conv2D)      (None, 14, 14, 96)   55296       ['block_11_depthwise_relu[0][0]']
                                                                                                      
     block_11_project_BN (BatchNorm  (None, 14, 14, 96)  384         ['block_11_project[0][0]']       
     alization)                                                                                       
                                                                                                      
     block_11_add (Add)             (None, 14, 14, 96)   0           ['block_10_project_BN[0][0]',    
                                                                      'block_11_project_BN[0][0]']    
                                                                                                      
     block_12_expand (Conv2D)       (None, 14, 14, 576)  55296       ['block_11_add[0][0]']           
                                                                                                      
     block_12_expand_BN (BatchNorma  (None, 14, 14, 576)  2304       ['block_12_expand[0][0]']        
     lization)                                                                                        
                                                                                                      
     block_12_expand_relu (ReLU)    (None, 14, 14, 576)  0           ['block_12_expand_BN[0][0]']     
                                                                                                      
     block_12_depthwise (DepthwiseC  (None, 14, 14, 576)  5184       ['block_12_expand_relu[0][0]']   
     onv2D)                                                                                           
                                                                                                      
     block_12_depthwise_BN (BatchNo  (None, 14, 14, 576)  2304       ['block_12_depthwise[0][0]']     
     rmalization)                                                                                     
                                                                                                      
     block_12_depthwise_relu (ReLU)  (None, 14, 14, 576)  0          ['block_12_depthwise_BN[0][0]']  
                                                                                                      
     block_12_project (Conv2D)      (None, 14, 14, 96)   55296       ['block_12_depthwise_relu[0][0]']
                                                                                                      
     block_12_project_BN (BatchNorm  (None, 14, 14, 96)  384         ['block_12_project[0][0]']       
     alization)                                                                                       
                                                                                                      
     block_12_add (Add)             (None, 14, 14, 96)   0           ['block_11_add[0][0]',           
                                                                      'block_12_project_BN[0][0]']    
                                                                                                      
     block_13_expand (Conv2D)       (None, 14, 14, 576)  55296       ['block_12_add[0][0]']           
                                                                                                      
     block_13_expand_BN (BatchNorma  (None, 14, 14, 576)  2304       ['block_13_expand[0][0]']        
     lization)                                                                                        
                                                                                                      
     block_13_expand_relu (ReLU)    (None, 14, 14, 576)  0           ['block_13_expand_BN[0][0]']     
                                                                                                      
     block_13_pad (ZeroPadding2D)   (None, 15, 15, 576)  0           ['block_13_expand_relu[0][0]']   
                                                                                                      
     block_13_depthwise (DepthwiseC  (None, 7, 7, 576)   5184        ['block_13_pad[0][0]']           
     onv2D)                                                                                           
                                                                                                      
     block_13_depthwise_BN (BatchNo  (None, 7, 7, 576)   2304        ['block_13_depthwise[0][0]']     
     rmalization)                                                                                     
                                                                                                      
     block_13_depthwise_relu (ReLU)  (None, 7, 7, 576)   0           ['block_13_depthwise_BN[0][0]']  
                                                                                                      
     block_13_project (Conv2D)      (None, 7, 7, 160)    92160       ['block_13_depthwise_relu[0][0]']
                                                                                                      
     block_13_project_BN (BatchNorm  (None, 7, 7, 160)   640         ['block_13_project[0][0]']       
     alization)                                                                                       
                                                                                                      
     block_14_expand (Conv2D)       (None, 7, 7, 960)    153600      ['block_13_project_BN[0][0]']    
                                                                                                      
     block_14_expand_BN (BatchNorma  (None, 7, 7, 960)   3840        ['block_14_expand[0][0]']        
     lization)                                                                                        
                                                                                                      
     block_14_expand_relu (ReLU)    (None, 7, 7, 960)    0           ['block_14_expand_BN[0][0]']     
                                                                                                      
     block_14_depthwise (DepthwiseC  (None, 7, 7, 960)   8640        ['block_14_expand_relu[0][0]']   
     onv2D)                                                                                           
                                                                                                      
     block_14_depthwise_BN (BatchNo  (None, 7, 7, 960)   3840        ['block_14_depthwise[0][0]']     
     rmalization)                                                                                     
                                                                                                      
     block_14_depthwise_relu (ReLU)  (None, 7, 7, 960)   0           ['block_14_depthwise_BN[0][0]']  
                                                                                                      
     block_14_project (Conv2D)      (None, 7, 7, 160)    153600      ['block_14_depthwise_relu[0][0]']
                                                                                                      
     block_14_project_BN (BatchNorm  (None, 7, 7, 160)   640         ['block_14_project[0][0]']       
     alization)                                                                                       
                                                                                                      
     block_14_add (Add)             (None, 7, 7, 160)    0           ['block_13_project_BN[0][0]',    
                                                                      'block_14_project_BN[0][0]']    
                                                                                                      
     block_15_expand (Conv2D)       (None, 7, 7, 960)    153600      ['block_14_add[0][0]']           
                                                                                                      
     block_15_expand_BN (BatchNorma  (None, 7, 7, 960)   3840        ['block_15_expand[0][0]']        
     lization)                                                                                        
                                                                                                      
     block_15_expand_relu (ReLU)    (None, 7, 7, 960)    0           ['block_15_expand_BN[0][0]']     
                                                                                                      
     block_15_depthwise (DepthwiseC  (None, 7, 7, 960)   8640        ['block_15_expand_relu[0][0]']   
     onv2D)                                                                                           
                                                                                                      
     block_15_depthwise_BN (BatchNo  (None, 7, 7, 960)   3840        ['block_15_depthwise[0][0]']     
     rmalization)                                                                                     
                                                                                                      
     block_15_depthwise_relu (ReLU)  (None, 7, 7, 960)   0           ['block_15_depthwise_BN[0][0]']  
                                                                                                      
     block_15_project (Conv2D)      (None, 7, 7, 160)    153600      ['block_15_depthwise_relu[0][0]']
                                                                                                      
     block_15_project_BN (BatchNorm  (None, 7, 7, 160)   640         ['block_15_project[0][0]']       
     alization)                                                                                       
                                                                                                      
     block_15_add (Add)             (None, 7, 7, 160)    0           ['block_14_add[0][0]',           
                                                                      'block_15_project_BN[0][0]']    
                                                                                                      
     block_16_expand (Conv2D)       (None, 7, 7, 960)    153600      ['block_15_add[0][0]']           
                                                                                                      
     block_16_expand_BN (BatchNorma  (None, 7, 7, 960)   3840        ['block_16_expand[0][0]']        
     lization)                                                                                        
                                                                                                      
     block_16_expand_relu (ReLU)    (None, 7, 7, 960)    0           ['block_16_expand_BN[0][0]']     
                                                                                                      
     block_16_depthwise (DepthwiseC  (None, 7, 7, 960)   8640        ['block_16_expand_relu[0][0]']   
     onv2D)                                                                                           
                                                                                                      
     block_16_depthwise_BN (BatchNo  (None, 7, 7, 960)   3840        ['block_16_depthwise[0][0]']     
     rmalization)                                                                                     
                                                                                                      
     block_16_depthwise_relu (ReLU)  (None, 7, 7, 960)   0           ['block_16_depthwise_BN[0][0]']  
                                                                                                      
     block_16_project (Conv2D)      (None, 7, 7, 320)    307200      ['block_16_depthwise_relu[0][0]']
                                                                                                      
     block_16_project_BN (BatchNorm  (None, 7, 7, 320)   1280        ['block_16_project[0][0]']       
     alization)                                                                                       
                                                                                                      
     Conv_1 (Conv2D)                (None, 7, 7, 1280)   409600      ['block_16_project_BN[0][0]']    
                                                                                                      
     Conv_1_bn (BatchNormalization)  (None, 7, 7, 1280)  5120        ['Conv_1[0][0]']                 
                                                                                                      
     out_relu (ReLU)                (None, 7, 7, 1280)   0           ['Conv_1_bn[0][0]']              
                                                                                                      
     global_average_pooling2d (Glob  (None, 1280)        0           ['out_relu[0][0]']               
     alAveragePooling2D)                                                                              
                                                                                                      
     dense (Dense)                  (None, 128)          163968      ['global_average_pooling2d[0][0]'
                                                                     ]                                
                                                                                                      
     dense_1 (Dense)                (None, 128)          16512       ['dense[0][0]']                  
                                                                                                      
     dense_2 (Dense)                (None, 101)          13029       ['dense_1[0][0]']                
                                                                                                      
    ==================================================================================================
    Total params: 2,451,493
    Trainable params: 193,509
    Non-trainable params: 2,257,984
    __________________________________________________________________________________________________
    None
    

## 모델 적용하기


```python
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(
    train_images,
    validation_data = val_images,
    epochs = 10,
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)
    ]
)
```

    Epoch 1/10
    531/531 [==============================] - 991s 2s/step - loss: 2.9299 - accuracy: 0.3060 - val_loss: 2.0081 - val_accuracy: 0.4833
    Epoch 2/10
    531/531 [==============================] - 1016s 2s/step - loss: 1.9425 - accuracy: 0.4985 - val_loss: 1.5583 - val_accuracy: 0.5865
    Epoch 3/10
    531/531 [==============================] - 896s 2s/step - loss: 1.6306 - accuracy: 0.5699 - val_loss: 1.3454 - val_accuracy: 0.6368
    Epoch 4/10
    531/531 [==============================] - 51782s 98s/step - loss: 1.3999 - accuracy: 0.6220 - val_loss: 1.0921 - val_accuracy: 0.7085
    Epoch 5/10
    531/531 [==============================] - 1046s 2s/step - loss: 1.1974 - accuracy: 0.6723 - val_loss: 0.9589 - val_accuracy: 0.7379
    Epoch 6/10
    531/531 [==============================] - 1004s 2s/step - loss: 1.0304 - accuracy: 0.7143 - val_loss: 0.8195 - val_accuracy: 0.7769
    Epoch 7/10
    531/531 [==============================] - 1040s 2s/step - loss: 0.8779 - accuracy: 0.7522 - val_loss: 0.6424 - val_accuracy: 0.8260
    Epoch 8/10
    531/531 [==============================] - 1013s 2s/step - loss: 0.7469 - accuracy: 0.7856 - val_loss: 0.5803 - val_accuracy: 0.8393
    Epoch 9/10
    531/531 [==============================] - 991s 2s/step - loss: 0.6202 - accuracy: 0.8221 - val_loss: 0.4754 - val_accuracy: 0.8696
    Epoch 10/10
    531/531 [==============================] - 979s 2s/step - loss: 0.5032 - accuracy: 0.8581 - val_loss: 0.3884 - val_accuracy: 0.8894
    

### 평가


```python
results = model.evaluate(test_images, verbose = 1) #verbose  상세한 정보 표준 출력 설정으로 0은 출력x,1은 자세히,2는 함축적 정보 
print("Precision: {:.2f}%".format(results[1]*100))
```

    285/285 [==============================] - 306s 1s/step - loss: 2.8072 - accuracy: 0.4657
    Precision: 46.57%
    

### 예측


```python
import requests
from PIL import Image
from io import BytesIO
import cv2

url = 'https://www.hola.com/imagenes/cocina/tecnicas-de-cocina/20210820194795/como-hacer-patatas-fritas-perfectas/0-986-565/portada-patatas-age-t.jpg'
response = requests.get(url)
test_img = Image.open(BytesIO(response.content))
test_img = np.array(test_img).astype(float)/255

test_img = cv2.resize(test_img, (224, 224))
predict = model.predict(test_img.reshape(-1,224,224,3))

print(np.argmax(predict))
```

    40
    


```python
# 클래스 카테고리 확인 

test_images.class_indices
```




    {'apple_pie': 0,
     'baby_back_ribs': 1,
     'baklava': 2,
     'beef_carpaccio': 3,
     'beef_tartare': 4,
     'beet_salad': 5,
     'beignets': 6,
     'bibimbap': 7,
     'bread_pudding': 8,
     'breakfast_burrito': 9,
     'bruschetta': 10,
     'caesar_salad': 11,
     'cannoli': 12,
     'caprese_salad': 13,
     'carrot_cake': 14,
     'ceviche': 15,
     'cheese_plate': 16,
     'cheesecake': 17,
     'chicken_curry': 18,
     'chicken_quesadilla': 19,
     'chicken_wings': 20,
     'chocolate_cake': 21,
     'chocolate_mousse': 22,
     'churros': 23,
     'clam_chowder': 24,
     'club_sandwich': 25,
     'crab_cakes': 26,
     'creme_brulee': 27,
     'croque_madame': 28,
     'cup_cakes': 29,
     'deviled_eggs': 30,
     'donuts': 31,
     'dumplings': 32,
     'edamame': 33,
     'eggs_benedict': 34,
     'escargots': 35,
     'falafel': 36,
     'filet_mignon': 37,
     'fish_and_chips': 38,
     'foie_gras': 39,
     'french_fries': 40,
     'french_onion_soup': 41,
     'french_toast': 42,
     'fried_calamari': 43,
     'fried_rice': 44,
     'frozen_yogurt': 45,
     'garlic_bread': 46,
     'gnocchi': 47,
     'greek_salad': 48,
     'grilled_cheese_sandwich': 49,
     'grilled_salmon': 50,
     'guacamole': 51,
     'gyoza': 52,
     'hamburger': 53,
     'hot_and_sour_soup': 54,
     'hot_dog': 55,
     'huevos_rancheros': 56,
     'hummus': 57,
     'ice_cream': 58,
     'lasagna': 59,
     'lobster_bisque': 60,
     'lobster_roll_sandwich': 61,
     'macaroni_and_cheese': 62,
     'macarons': 63,
     'miso_soup': 64,
     'mussels': 65,
     'nachos': 66,
     'omelette': 67,
     'onion_rings': 68,
     'oysters': 69,
     'pad_thai': 70,
     'paella': 71,
     'pancakes': 72,
     'panna_cotta': 73,
     'peking_duck': 74,
     'pho': 75,
     'pizza': 76,
     'pork_chop': 77,
     'poutine': 78,
     'prime_rib': 79,
     'pulled_pork_sandwich': 80,
     'ramen': 81,
     'ravioli': 82,
     'red_velvet_cake': 83,
     'risotto': 84,
     'samosa': 85,
     'sashimi': 86,
     'scallops': 87,
     'seaweed_salad': 88,
     'shrimp_and_grits': 89,
     'spaghetti_bolognese': 90,
     'spaghetti_carbonara': 91,
     'spring_rolls': 92,
     'steak': 93,
     'strawberry_shortcake': 94,
     'sushi': 95,
     'tacos': 96,
     'takoyaki': 97,
     'tiramisu': 98,
     'tuna_tartare': 99,
     'waffles': 100}



==> 해당 url의 사진은 감자튀김이였고 클래스 번호를 보면 40번은 french_fries로 맞춘 것을 확인할 수 있음
