images = np.array(images)
labels = np.array(labels)
defect_classes = ['Grade-1','Grade-2','Grade-3']
num_of_classes = 3
classes_encoded = to_categorical(labels,num_classes = num_of_classes)
images = images/255.0
X_train,X_test,y_train,y_test = train_test_split(images,classes_encoded,test_size=0.2,random_state=10)
X_train.shape
X_test.shape
