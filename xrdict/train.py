from xrdict.evaluate import evaluate
from tqdm import tqdm
import torch
import gc


def train(model, n_epochs, data, optimizer, batch_size, device):

    model.to(device)

    train_dataset, val_dataset, test_dataset = data
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=val_dataset.collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    best_val_acc = 0.
    for epoch in range(n_epochs):
        print(f'Running epoch {epoch+1}...')
        train_loss, train_labels, train_preds = 0, [], []

        for target_ids, def_ids, lens in tqdm(train_dataloader):
            target_ids, def_ids, lens = target_ids.to(device), def_ids.to(device), lens.to(device)
            model.train()
            optimizer.zero_grad()
            loss, _, indices = model(x=def_ids, lengths=lens, word_gt=target_ids, mode='train')
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            predicted = indices[:, :100].detach().cpu().numpy().tolist()
            train_loss += loss.item()
            train_labels.extend(target_ids.detach().cpu().numpy().tolist())
            train_preds.extend(predicted)

        train_acc_1, train_acc_10, train_acc_100 = evaluate(train_labels, train_preds)
        del train_labels, train_preds, lens
        gc.collect()

        print('train_loss: ', train_loss/len(train_dataset))
        print('train_acc(1/10/100): %.2f %.2f %.2f\n' % (train_acc_1, train_acc_10, train_acc_100))

        model.eval()
        with torch.no_grad():
            print(f'Validating epoch {epoch+1}...')
            val_labels, val_preds = [], []
            for target_ids, def_ids, lens in tqdm(val_dataloader):
                target_ids, def_ids, lens = target_ids.to(device), def_ids.to(device), lens.to(device)
                _, indices = model(x=def_ids, lengths=lens, mode='test')
                predicted = indices[:, :100].detach().cpu().numpy().tolist()
                val_labels.extend(target_ids.detach().cpu().numpy().tolist())
                val_preds.extend(predicted)

            val_acc_1, val_acc_10, val_acc_100 = evaluate(val_labels, val_preds)
            print('valid_acc(1/10/100): %.2f %.2f %.2f\n' % (val_acc_1, val_acc_10, val_acc_100))
            del val_labels, val_preds, lens
            gc.collect()

            if val_acc_10 > best_val_acc:
                print('*BEST VALID ACCURACY*')
                print(f'Testing epoch {epoch+1}...')
                best_val_acc = val_acc_10
                # torch.save(model, f'../saves/model_{epoch+1}.pth')
                
                test_labels, test_preds = [], []
                for target_ids, def_ids, lens in tqdm(test_dataloader):
                    target_ids = target_ids.to(device)
                    def_ids = def_ids.to(device)
                    lens = lens.to(device)
                    _, indices = model(x=def_ids, lengths=lens, mode='test')
                    predicted = indices[:, :100].detach().cpu().numpy().tolist()
                    test_labels.extend(target_ids.detach().cpu().numpy().tolist())
                    test_preds.extend(predicted)
                
                test_acc_1, test_acc_10, test_acc_100 = evaluate(test_labels, test_preds)
                print('test_acc(1/10/100): %.2f %.2f %.2f\n' % (test_acc_1, test_acc_10, test_acc_100))
                del test_labels, test_preds, lens
                gc.collect()
        
    return model
