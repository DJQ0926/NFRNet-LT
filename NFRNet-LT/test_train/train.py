from config import *
from utils import *
# from model import * #3
from model2 import *    #4
from focal_loss import *    #focal loss
import torch.optim as optim
from EML_focalloss import *  
from torch.optim.lr_scheduler import LambdaLR


def lr_lambda(epoch):
    return 0.1 if epoch == 15 else 1


if __name__ == '__main__':
    id2label,_= get_label()
    train_dataset = Dataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    # dev_dataset = Dataset('dev')
    # dev_loader = data.DataLoader(dev_dataset, batch_size=128, shuffle=False)

 #model
    model = TextCNN().to(DEVICE)

 #optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR,weight_decay=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=LR,weight_decay=0.001)

# loss
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = FocalLossWithL2Reg(gamma=2)
    loss_fn = MultiLossFocalLoss(alpha=0.25, gamma=2, reduction='mean', num_classes=NUM_CLASSES, beta=0.98)
    # loss_fn =FocalLoss(gamma=2)
    scheduler = LambdaLR(optimizer, lr_lambda)
    print('Start training:')

# epoch——training
    for e in range(EPOCH):
        for b, (input, mask, target) in enumerate(train_loader):
            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)
            pred = model(input, mask)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  

            if b % 50 != 0:
                continue

            y_pred = torch.argmax(pred, dim=1)
            report = evaluate(y_pred.cpu().data.numpy(), target.cpu().data.numpy(), output_dict=True)
            

            # with torch.no_grad():
            #     dev_input, dev_mask, dev_target = iter(dev_loader).next()
            #     dev_input = dev_input.to(DEVICE)
            #     dev_mask = dev_mask.to(DEVICE)
            #     dev_target = dev_target.to(DEVICE)
            #     dev_pred = model(dev_input, dev_mask)
            #     dev_pred_ = torch.argmax(dev_pred, dim=1)
            #     dev_report = evaluate(dev_pred_.cpu().data.numpy(), dev_target.cpu().data.numpy(), output_dict=True)

            print(
                '>> epoch:', e,
                'batch:', b,
                'loss:', round(loss.item(), 5),
                'train_acc:', report['accuracy'],
                # 'dev_acc:', dev_report['accuracy'],
            )

    # torch.save(model, MODEL_DIR1 + f'test_{e}.pth')
    torch.save(model, '/path'+ f'model{e}.pth')