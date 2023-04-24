from preprocessing import Integer, Categorical, Dense,\
    Sequential, BayesSearchCV, KerasClassifier, X_train, y_train
# Step 4:  Define the search space for the hyperparameters
search_space = {
    'neurons': Integer(8, 64),
    'hidden_layers': Integer(1, 4),
    'optimizer': Categorical(['adam', 'sgd']),
    'activation': Categorical(['relu', 'sigmoid', 'tanh'])
}

# Define the model to be optimized
def build_model(neurons=8, hidden_layers=1, optimizer='sgd', activation='relu'):
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_shape=(17,)))
    for i in range(hidden_layers):
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Define the Bayesian search
bayes_cv = BayesSearchCV(
    estimator=KerasClassifier(build_fn=build_model, epochs=50, batch_size=32,
                              activation='relu', optimizer='sgd', hidden_layers=1, neurons=8),
    search_spaces=search_space,
    n_iter=20,
    cv=3,
    n_jobs=-1,
    verbose=1
)

# Run the Bayesian search
bayes_cv.fit(X_train, y_train)

# Print the best hyperparameters and accuracy
print("Best hyperparameters:", bayes_cv.best_params_)
print("Best accuracy:", bayes_cv.best_score_)