import os
import gc
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from tqdm import tqdm
import torch
from jitvul_model.graph_dataset import *
from jitvul_model.Clasifier import *
import pandas

tqdm.pandas()


def train_model(
    train_file_path, encoder, tokenizer, _params, model_path, starting_epochs=0
):
    torch.manual_seed(12345)

    train_dataset = GraphDataset(train_file_path, tokenizer, _params)
    _trainLoader = DataLoader(train_dataset, collate_fn=collate_batch, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epochs = _params["max_epochs"]
    data = {}
    model = Clasifier(encoder=encoder, dropout=_params["dropout_rate"])

    model.to(device)
    print(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=_params["lr"], betas=(0.9, 0.999), eps=1e-08
    )
    print("learning rate : ", optimizer.param_groups[0]["lr"])
    criterion = nn.CrossEntropyLoss()
    starting_epochs += 1
    valid_auc = 0
    last_train_loss = -1
    last_acc = 0
    for e in range(starting_epochs, max_epochs):
        train_loss, acc = train(e, _trainLoader, model, criterion, optimizer, device)
        if last_train_loss == -1 or last_train_loss > train_loss:
            saved_model_path = os.path.join(
                os.path.join(os.getcwd(), model_path), _params["model_name"] + ".pt"
            )
            torch.save(model.state_dict(), saved_model_path)
            last_train_loss = train_loss
        gc.collect()


def train(curr_epochs, _trainLoader, model, criterion, optimizer, device):
    train_loss = 0
    correct = 0
    model.train()
    for input_ids, attn, label, commit_id in _trainLoader:
        if device != "cpu":
            input_ids = input_ids.cuda()
            attn = attn.cuda()
            label = label.cuda()
        target = label
        out = model(input_ids, attn)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = out.max(1)
        correct += predicted.eq(target).sum().item()
        del input_ids, attn, label, predicted, out
    avg_train_loss = train_loss / len(_trainLoader)
    acc = correct / len(_trainLoader)
    print("correct:", correct)
    print(
        "epochs {}".format(curr_epochs)
        + " train loss: {}".format(avg_train_loss)
        + " acc: {}".format(acc)
    )
    gc.collect()
    return avg_train_loss, acc


def test_model(test_file_path, encoder, tokenizer, _params, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = GraphDataset(test_file_path, tokenizer, _params)
    _testLoader = DataLoader(test_dataset, collate_fn=collate_batch, shuffle=False)
    test_model = Clasifier(encoder=encoder, dropout=_params["dropout_rate"])

    test_model.load_state_dict(
        torch.load(
            os.path.join(
                os.path.join(os.getcwd(), model_path), _params["model_name"] + ".pt"
            )
        )
    )
    test_model.eval()
    evaluate_metrics(_params["model_name"], test_model, _testLoader, device)


def evaluate_metrics(model_name, model, _loader, device):
    print("evaluate >")
    write_to_file_results = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        all_predictions, all_targets, all_probs = [], [], []
        for input_ids, attn, label, commit_id in _loader:
            if device != "cpu":
                input_ids = input_ids.cuda()
                attn = attn.cuda()
                label = label.cuda()
            target = label
            out = model(input_ids, attn)
            target = target.cpu().detach().numpy()
            pred = out.argmax(dim=1).cpu().detach().numpy()
            prob_1 = out.cpu().detach().numpy()[0][1]
            write_to_file_results.append(
                {"commit_id": commit_id, "predict": pred[0], "target": graph.y.item()}
            )
            all_probs.append(prob_1)
            all_predictions.append(pred)
            all_targets.append(target)
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        auc_score = round(auc(fpr, tpr) * 100, 2)
        acc = round(accuracy_score(all_targets, all_predictions) * 100, 2)
        print(acc)
        precision = round(precision_score(all_targets, all_predictions) * 100, 2)
        f1 = round(f1_score(all_targets, all_predictions) * 100, 2)
        recall = round(recall_score(all_targets, all_predictions) * 100, 2)
        matrix = confusion_matrix(all_targets, all_predictions)
        target_names = ["clean", "buggy"]
        report = classification_report(
            all_targets, all_predictions, target_names=target_names
        )
        result = (
            "auc: {}".format(auc_score)
            + " acc: {}".format(acc)
            + " precision: {}".format(precision)
            + " recall: {}".format(recall)
            + " f1: {}".format(f1)
            + " \nreport:\n{}".format(report)
            + "\nmatrix:\n{}".format(matrix)
        )
        print(result)
    df = pandas.DataFrame.from_dict(write_to_file_results)
    df.to_csv(r"Data/result/" + model_name + ".csv", index=True, header=True)
    model.train()
