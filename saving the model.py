import pickle
filename = 'saved_model.sav'
pickle.dump(defect_classifier,open(filename,'wb'))
model = pickle.load(open('saved_model.sav','rb'))
prediction = model.predict(input_horizontal)
predicted_class = defect_classes[np.argmax(prediction,axis=1).item()]
print(f"Prediction of class: {predicted_class}")
