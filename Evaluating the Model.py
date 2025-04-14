defect_classifier.evaluate(X_test,y_test)
y_predictions = defect_classifier.predict(X_test)
y_predictions = [np.argmax(i) for i in y_predictions]
y_actual = [np.argmax(i) for i in y_test]
cm = confusion_matrix(y_actual,y_predictions)
plt.figure(figsize=(10,7))
sb.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Prediction')
plt.ylabel('True')
plt.show()
print(classification_report(y_actual,y_predictions))
