from preprocessing import Sequential, Dense, X_train, y_train, X_test, y_test, classification_report

# Step 5: Build the neural network - (this model is based on results from Grid Search)
model = Sequential()
model.add(Dense(17, activation='tanh', input_shape=(17,)))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Step 6: Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 7: Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(X_test, y_test), verbose=1)

# Step 8: Evaluate the model
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)  # convert probabilities to binary class labels
print(classification_report(y_test, y_pred))

# Step 9 (optional): Plot loss, accuracy, confusion matrix, ROC curve (see figures.py)
