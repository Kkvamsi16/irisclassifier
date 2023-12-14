import sklearn
print(sklearn.__version__)
from sklearn.ensemble import RandomForestClassifier
import pickle

# Assuming model creation and training process
model = RandomForestClassifier()
# Train your model here...

# Save the model with the updated scikit-learn version
with open('savedmodel.sav', 'wb') as file:
    pickle.dump(model, file)
