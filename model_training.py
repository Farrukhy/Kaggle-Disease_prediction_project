This script contains the model training process using RandomForestClassifier and 
GradientBoostingClassifier for disease prediction,
and it evaluates the model using accuracy scores.


  from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Train RandomForest Model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_imputed, y_train)
y_pred_rf = model_rf.predict(X_test_imputed)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf:.4f}')

# Train Gradient Boosting Model
classifier_gb = GradientBoostingClassifier(n_estimators=100)
classifier_gb.fit(X_train_imputed, y_train)
y_pred_gb = classifier_gb.predict(X_test_imputed)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f'Gradient Boosting Accuracy: {accuracy_gb:.4f}')
