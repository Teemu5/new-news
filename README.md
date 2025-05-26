# TRAIN DEFAULT MODELS WITH COMMAND: python3 train_all_models.py
# EVAL DEFAULT MODEL WITH COMMAND: python3 eval_all_models.py
# TRAIN DEFAULT META MODEL WITH COMMAND: python3 eval_all_meta_models.py

Or do it all with:
cd backend;
python3 train_all_models.py
python3 eval_all_models.py
python3 eval_all_meta_models.py

# Meta models' results are found at backend/meta/results/ in .json files
# Base models' results are found at backend/meta/base_preds/ in .json files