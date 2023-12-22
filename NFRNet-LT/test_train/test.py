from config import *
from utils import *
# from model import *
from model2 import *
from EML_focalloss import *
from focal_loss import *
import numpy as np
import csv
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score


def top_k_accuracy(y_true, y_pred, k=5):
    correct = 0
    total = len(y_true)
    
    for i in range(total):
        if y_true[i] in y_pred[i][:k]:
            correct += 1

    return correct / total

if __name__ == '__main__':
   # Get label mapping
    id2label, _ = get_label()
    # Load test set data
    test_dataset = Dataset('test')
    test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

#    Loading the trained model
    model = torch.load('/path').to(DEVICE)
    
    loss_fn = MultiLossFocalLoss(alpha=0.25, gamma=2, reduction='mean', num_classes=NUM_CLASSES, beta=0.98)
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn =FocalLoss(gamma=2)
    y_pred = []
    y_true = []
    top5_pred= []
    N = len(y_true)

    with torch.no_grad():
     
        for b, (input, mask, target) in enumerate(test_loader):
            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)
            test_pred = model(input, mask)
            loss = loss_fn(test_pred, target)

  
            print('>> batch:', b, 'loss:', round(loss.item(), 5))
            

            # top-5
            top5_pred_ = torch.topk(test_pred, k=5, dim=1)[1].cpu().numpy()
            top5_pred.extend(top5_pred_)

        
            test_pred_ = torch.argmax(test_pred, dim=1)
            y_pred += test_pred_.data.tolist()

            y_true += target.data.tolist()


# top-1  accuracy
acc_top1 = accuracy_score(y_true, y_pred)

# top-5  accuracy
acc_top5 = top_k_accuracy(y_true, top5_pred, k=5)


y_true = np.array(y_true)
top5_pred = np.array(top5_pred)

for i in range(NUM_CLASSES):

    label_index = i
    label_name = id2label[i]

 
    label_indices = np.where(y_true == label_index)[0]
    label_top5_preds = top5_pred[label_indices]


    label_true_labels = y_true[label_indices]  # Add this line
    label_top5_accuracy = top_k_accuracy(y_true=label_true_labels, y_pred=label_top5_preds, k=5)  # Modify this line


    print(f"Label: {label_name}")
    print(f"Top-5 Accuracy: {round(label_top5_accuracy, 5)} (Support: {len(label_indices)})")


print("test results ：")
print(evaluate(y_pred, y_true, id2label))
print('>> top-1 accuracy:', round(acc_top1, 5))

print('>> top-5 accuracy:', round(acc_top5, 5))


