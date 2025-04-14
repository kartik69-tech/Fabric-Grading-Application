img_grade1 = cv2.imread('/content/drive/MyDrive/Fabric project/Fabric-Defect-Classification_CNN-main/Dataset/TestImages/test1.jpg')
resized_grade1= tf.image.resize(img_grade1,(224,224))
plt.imshow(resized_grade1.numpy().astype(int))
img_grade2 = cv2.imread('/content/drive/MyDrive/Fabric project/Fabric-Defect-Classification_CNN-main/Dataset/TestImages/test2.jpg')
resized_grade2 = tf.image.resize(img_grade2,(224,224))
plt.imshow(resized_grade2.numpy().astype(int))
img_grade3 = cv2.imread('/content/drive/MyDrive/Fabric project/Fabric-Defect-Classification_CNN-main/Dataset/TestImages/test3.jpg')
resized_grade3 = tf.image.resize(img_grade3,(224,224))
plt.imshow(resized_grade3.numpy().astype(int))
resized_grade1 = resized_grade1/255.0
resized_grade2 = resized_grade2/255.0
resized_grade3 = resized_grade3/255.0
input_grade1 = tf.expand_dims(resized_grade1,axis=0)
input_grade2 = tf.expand_dims(resized_grade2,axis=0)
input_grade3 = tf.expand_dims(resized_grade3,axis=0)
prediction_1 = defect_classifier.predict(input_grade1)
prediction_2 = defect_classifier.predict(input_grade2)
prediction_3 = defect_classifier.predict(input_grade3)

predicted_class1 = np.argmax(prediction_1,axis=1).item()
predicted_class2 = np.argmax(prediction_2,axis=1).item()
predicted_class3 = np.argmax(prediction_3,axis=1).item()
predicted_label1 = defect_classes[predicted_class1]
predicted_label2 = defect_classes[predicted_class2]
predicted_label3 = defect_classes[predicted_class3]

print(f"Prediction of first image: {predicted_label1}")
print(f"Prediction of second image: {predicted_label2}")
print(f"Prediction of third image: {predicted_label3}")
