from preprocessing import Sequential, Dense, \
    KerasClassifier, GridSearchCV, X_train, y_train


# Step 4. Grid search for optimal parameters

def create_model(neurons=8, hidden_layers=1, optimizer='sgd', activation='relu'):
    model = Sequential()
    for i in range(hidden_layers):
        model.add(Dense(neurons, input_dim=17, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, verbose=0, hidden_layers=1,
                        neurons=8, activation='relu', optimizer='sgd')
param_grid = {
    'neurons': [8, 16, 32, 64],
    'hidden_layers': [1, 2, 3, 4],
    'optimizer': ['adam', 'sgd'],
    'activation': ['relu', 'sigmoid', 'tanh']
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_result = grid.fit(X_train, y_train)

print("Best parameters: ", grid_result.best_params_)
print("Best score: ", grid_result.best_score_)
