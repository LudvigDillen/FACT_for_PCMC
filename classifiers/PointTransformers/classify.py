def classify_pairs(model, points):
    classifier = model.eval()
    pred = classifier(
        points, inference=True
    )  # [B, n_classes], here we have a score for each class
    return pred.data.max(1)[1]  # highest score wins
