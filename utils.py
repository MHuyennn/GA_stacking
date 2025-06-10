def predict_pretrained(data, model_name, img_size=299, on_test_set=True, pretrained_dir=None):
    if pretrained_dir is None:
        raise ValueError("pretrained_dir must be provided for predict_pretrained.")
    print(f"Predicting on {'test' if on_test_set else 'val'} dataset...")
    dataset = data.x_test if on_test_set else data.x_val

    checkpoint_path = os.path.join(pretrained_dir, f"best_{MAPPING[model_name]}.pkl")
    print(f"Loading model {model_name} from {checkpoint_path}...")

    if model_name in DL_MODELS:
        model = load_model(checkpoint_path)
        y_pred = model.predict(dataset, batch_size=32)
    elif model_name in ML_MODELS:
        with open(checkpoint_path, 'rb') as f:
            model = pickle.load(f)
        # Reshape dataset to 2D: (num_samples, num_features)
        dataset_2d = dataset.reshape([-1, np.prod((img_size, img_size, 1))])
        y_pred = model.predict_proba(dataset_2d)  # Lấy toàn bộ ma trận xác suất
        # Nếu chỉ cần xác suất lớp dương (class 1), lấy cột thứ 1: y_pred[:, 1]
        # Nhưng cần đảm bảo model được huấn luyện cho binary classification

    return y_pred
