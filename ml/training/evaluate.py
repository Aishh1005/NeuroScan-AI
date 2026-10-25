"""
Evaluates a trained checkpoint on dataset/test/ and writes REAL metrics into
the application database (model_metrics table) so the Model Performance page
can display them. Nothing here is fabricated -- if you have not trained a
model yet, this script will simply fail to find a checkpoint and exit.

Usage:
    cd ml/training
    python evaluate.py --data-dir ../../dataset --arch resnet50
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
from app.ml.model_arch import get_model  # noqa: E402
from app.database.session import Base, engine, SessionLocal  # noqa: E402
from app.models.db_models import ModelMetric  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../../dataset")
    parser.add_argument("--arch", type=str, default="resnet50", choices=["resnet50", "efficientnet"])
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Defaults to ../../backend/models/classifier_<arch>.pt")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or f"../../backend/models/classifier_{args.arch}.pt"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found at {checkpoint_path}. Train a model first with train.py.")
        sys.exit(1)

    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.isdir(test_dir):
        print(f"ERROR: test directory not found at {test_dir}.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)
    if test_ds.classes != class_names:
        print("ERROR: test/ class folders do not match the classes the model was trained on.")
        sys.exit(1)

    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = get_model(architecture=args.arch, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    try:
        roc_auc = roc_auc_score(all_labels, np.array(all_probs), multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = None

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc}")
    print(f"Confusion matrix:\n{np.array(cm)}")

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.add(ModelMetric(
            model_name=f"classifier_{args.arch}",
            architecture=args.arch,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            roc_auc=roc_auc,
            confusion_matrix=json.dumps({"labels": class_names, "matrix": cm}),
            num_classes=len(class_names),
            train_dataset=os.path.abspath(args.data_dir),
            test_dataset=os.path.abspath(test_dir),
        ))
        db.commit()
        print("Saved real evaluation metrics to the database (model_metrics table).")
    finally:
        db.close()


if __name__ == "__main__":
    main()
