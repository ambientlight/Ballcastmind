## Target
Develop the supervised deep learning model that can estimate the camera parameters good enough so that the camera tracking algorithm can kick off.
This essentially replaces the manual human fitting of initial camera position for venue that has never been present before so no previous camara positions are available. 

The necessary requirement is that the actual recongized lines will be present within the project-lines windows so that the minimization process can be applied. Measure of fit also needs to develop to understand how good is the estimated camera location. So that in case fitValue falls below certain treshold, we can brute-force certain range of camera parameters for a better initial fit.

## Features
For the sake of understand what features can be more effectively be used as an input to our model, current the following seperation is taken into consideration:

* `raw`: raw image pixels
* `min-line-diff`: filtered image obtained after applying the line filter implementation from paper
* `line-detector-detected`: using line detector with detected lines drawn in clear image
* `line-detector-reduced`: using our line detector with reduction algorithm being applied before redrawing in clear image
* `lines-detected`: detected line coordinates [x1,y1,x2,y2] normalized by image size
* `lines-reduced`: detected lines and reduced by our algorithm, then normalized by image size

## Experimentation
To establish some raw baseline would might work - around 1000 frames from main camera has been anotatated from starting 3 mins of Liverpool - Hoffenhaim game. 

### Raw(rot-x): SeparableConv2D(4x)-Dense:adam
The best result so far was achieved by 4 sepable convolution layers staked before dense layer applied with following parameters:

```py
img_input = Input(shape=(1384, 865, 3), dtype='float32')
x = layers.SeparableConv2D(256, 20, strides=(10, 10), activation='relu')(img_input)
x = layers.SeparableConv2D(512, 7, strides=(2, 2), activation='relu')(x)
x = layers.SeparableConv2D(1024, 7, strides=(2, 2), activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
rot_y_pred = layers.Dense(1, name='rot_y')(x)

model = Model(img_input, rot_y_pred)
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])
```
The model can succesfully overfit to 0.01 degree, with validation accuracy near to 1 degree at best for now.

### Lines Reduced(rot-x): Dense-Dropout:adam
Reduced lines are able to achieve better benchmarks on a very simple single dense layer network with dropout. In particular, 0.35 validation loss on the following configuration. This is not so reliable though, since val-loss is lower then test loss in certain cases(overfitting test set) with large variance in epoch-to-epoch results, due to very small train/val data at this stage of expirement.

```py
line_input = Input(shape=(32, 4), dtype='float32')
x = layers.Dense(1024, activation='relu')(line_input)
x = layers.Dropout(0.5)(x)
x = layers.Flatten()(x)
rot_x_pred = layers.Dense(1, name='rot_x')(x)
model = Model(line_input, rot_x_pred)
model.compile(optimizer='adam',
              loss='mse')
```

### Min-line-diff(rot-x): SeparableConv2D(4x)-Dense:adam
The same architecture that we used for raw images achieved a train loss of 0.2 and validation loss 0.87 at best. 

### Line Detector Reduced(rot-x): SeparableConv2D(4x)-Dense:adam
The same architecture that we used for raw images achieved a train loss of 0.1 and validation loss ~0.5 at best. 

