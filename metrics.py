import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_text_acc(preds, labels):
    return np.mean(np.array(preds) == np.array(labels))


def compute_equation_acc(preds, labels):
    preds = [eval_equation(pred) for pred in preds]
    labels = [eval_equation(label) for label in labels]

    return np.mean(np.array(preds) == np.array(labels))


def eval_equation(equation):
    try:
        answer = eval(equation)
    except:
        answer = np.nan

    return answer


def compute_metrics_text(tokenizer, args):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions[0], skip_special_tokens=True)

        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        label_mapping = {'benign': 0, 'vulnerable': 1}
        numeric_labels = np.array([label_mapping[label] for label in decoded_labels])
        numeric_predictions = np.array([label_mapping.get(pred, -1) for pred in decoded_preds])

        TP = np.sum((numeric_predictions == 1) & (numeric_labels == 1))
        FP = np.sum((numeric_predictions == 1) & (numeric_labels == 0))
        FN = np.sum((numeric_predictions == 0) & (numeric_labels == 1))
        TN = np.sum((numeric_predictions == 0) & (numeric_labels == 0))


        # Accuracy, Precision, Recall, F1 Score
        accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"\npretrained_model: {args.from_pretrained.split('/')[-1]}; model_type: {args.model_type}; dataset_name: {args.dataset}, small_sample: {args.small_sample if args.small_sample != '' else 'all'}")
        print(f"accuracy:{accuracy*100:.4f}%")
        print(f"f1:{f1*100:.4f}%")
        print(f"precision:{precision*100:.4f}%")
        print(f"recall:{recall*100:.4f}%", flush=True)  # 最后一个刷新输出


        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

    return compute_metrics


def compute_metrics_text_aux(tokenizer, args):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        label_mapping = {'benign': 0, 'vulnerable': 1}
        numeric_labels = np.array([label_mapping[label] for label in decoded_labels])
        numeric_predictions = np.array([label_mapping.get(pred, -1) for pred in decoded_preds])

        TP = np.sum((numeric_predictions == 1) & (numeric_labels == 1))
        FP = np.sum((numeric_predictions == 1) & (numeric_labels == 0))
        FN = np.sum((numeric_predictions == 0) & (numeric_labels == 1))
        TN = np.sum((numeric_predictions == 0) & (numeric_labels == 0))


        # Accuracy, Precision, Recall, F1 Score
        accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"\npretrained_model: {args.from_pretrained.split('/')[-1]}; model_type: {args.model_type}; dataset_name: {args.dataset}, small_sample: {args.small_sample if args.small_sample != '' else 'all'}")
        print(f"accuracy:{accuracy*100:.4f}%")
        print(f"f1:{f1*100:.4f}%")
        print(f"precision:{precision*100:.4f}%")
        print(f"recall:{recall*100:.4f}%", flush=True)  # 最后一个刷新输出


        return {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

    return compute_metrics



def compute_metrics_equation(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions[0], skip_special_tokens=True)

        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds = list()
        for pred in decoded_preds:    
            preds.append(eval_equation(pred))

        labels = list()
        for label in decoded_labels:    
            labels.append(eval_equation(label))

        acc = np.mean(np.array(preds) == np.array(labels))

        return {'accuracy': acc}
    
    return compute_metrics


def compute_metrics_equation_aux(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds = list()
        for pred in decoded_preds:    
            preds.append(eval_equation(pred))

        labels = list()
        for label in decoded_labels:    
            labels.append(eval_equation(label))

        acc = np.mean(np.array(preds) == np.array(labels))

        return {'accuracy': acc}
    
    return compute_metrics
