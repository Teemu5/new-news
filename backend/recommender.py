import numpy as np
#from utils import tokenize_input
def tokenize_input(input_text: str):
    """
    Dummy preprocessing to transform raw input text into the 
    required input shape for Keras ensemble models.
    !! Replace this with your actual text preprocessing. !!
    """
    return np.array([[len(input_text), 1]])

def fastformer_model_predict(history_tensor, candidate_tensor, model):
    preds = model.predict({
    "history_input": history_tensor,
    "candidate_input": candidate_tensor
})
    return preds[0]

def ensemble_bagging(history_tensor, candidate_tensor, models: dict) -> np.ndarray:
    y1 = fastformer_model_predict(history_tensor, candidate_tensor, models[0])
    y2 = fastformer_model_predict(history_tensor, candidate_tensor, models[1])
    y3 = fastformer_model_predict(history_tensor, candidate_tensor, models[2])
    predictions = np.vstack([y1, y2, y3])
    return np.mean(predictions, axis=0), predictions

def ensemble_boosting(history_tensor, candidate_tensor, models: dict, errors: np.ndarray) -> np.ndarray:
    y1 = fastformer_model_predict(history_tensor, candidate_tensor, models[0])
    y2 = fastformer_model_predict(history_tensor, candidate_tensor, models[1])
    y3 = fastformer_model_predict(history_tensor, candidate_tensor, models[2])
    predictions = np.vstack([y1, y2, y3])
    errors = np.where(errors == 0, 1e-6, errors)
    weights = 1 / errors
    weights = weights / np.sum(weights)
    return np.average(predictions, axis=0, weights=weights)

def train_stacking_meta_model(X_train: np.ndarray, y_train: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    meta_model = LogisticRegression()
    meta_model.fit(X_train, y_train)
    return meta_model

def ensemble_stacking(history_tensor, candidate_tensor, models: dict, meta_model) -> np.ndarray:
    y1 = fastformer_model_predict(history_tensor, candidate_tensor, models[0])
    y2 = fastformer_model_predict(history_tensor, candidate_tensor, models[1])
    y3 = fastformer_model_predict(history_tensor, candidate_tensor, models[2])
    X = np.vstack([y1, y2, y3]).T  # each column is one model's prediction
    final_predictions = meta_model.predict_proba(X)[:, 1]
    return final_predictions

def hybrid_ensemble(history_tensor, candidate_tensor, models: dict, boosting_errors: np.ndarray, stacking_meta_model) -> np.ndarray:
    bagging_pred = ensemble_bagging(history_tensor, candidate_tensor, models)
    boosting_pred = ensemble_boosting(history_tensor, candidate_tensor, models, boosting_errors)
    stacking_pred = ensemble_stacking(history_tensor, candidate_tensor, models, stacking_meta_model)
    final_prediction = (bagging_pred + boosting_pred + stacking_pred) / 3
    return final_prediction
