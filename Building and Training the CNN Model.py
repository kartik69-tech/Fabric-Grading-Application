defect_classifier = Sequential()

#Adding the CNN layers
defect_classifier.add(Conv2D(16,(3,3),1,activation='relu',input_shape=(224,224,3)))
defect_classifier.add(MaxPooling2D(2,2))
defect_classifier.add(Conv2D(32,(3,3),1,activation='relu'))
defect_classifier.add(MaxPooling2D(2,2))
defect_classifier.add(Conv2D(64,(3,3),1,activation='relu'))
defect_classifier.add(MaxPooling2D(2,2))
defect_classifier.add(Conv2D(32,(3,3),1,activation='relu'))
defect_classifier.add(MaxPooling2D())
defect_classifier.add(Conv2D(16,(3,3),1,activation='relu'))
defect_classifier.add(MaxPooling2D())

#Flattening the output of Conv
defect_classifier.add(Flatten())

#Adding Hidden layers along with Dropout
defect_classifier.add(Dense(128,activation='relu'))
defect_classifier.add(Dropout(0.25))
defect_classifier.add(Dense(64,activation='relu'))
defect_classifier.add(Dropout(0.2))

#Output layer
defect_classifier.add(Dense(3,activation='softmax'))

defect_classifier.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
filepath = '/content/model.keras'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
callbacks = [checkpoint]
defect_classifier.summary()
