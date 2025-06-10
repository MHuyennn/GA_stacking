from __init__ import *

def predict_pretrained(data, model_name, img_size=299, on_test_set=True, pretrained_dir=None):
    if pretrained_dir is None:
        raise ValueError("pretrained_dir must be provided for predict_pretrained.")
    print(f"Predicting on {'test' if on_test_set else 'val'} dataset...")
    dataset = data.x_test if on_test_set else data.x_val
    checkpoint_path = os.path.join(pretrained_dir, f"best_{MAPPING[model_name]}.{'keras' if model_name in DL_MODELS else 'pkl'}")
    print(f"Loading model {model_name} from {checkpoint_path}...")
    if model_name in DL_MODELS:
        model = load_model(checkpoint_path)
        y_pred = model.predict(dataset, batch_size=32)
    elif model_name in ML_MODELS:
        with open(checkpoint_path, 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict_proba(dataset.reshape([-1, np.prod((img_size, img_size, 1))])[:, 1])  # Thêm dấu ')' bị thiếu
    return y_pred