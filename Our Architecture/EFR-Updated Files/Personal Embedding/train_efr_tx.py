import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils import data
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

batch_size = 128
seq_len = 5
seq2_len = seq_len
emb_size = 768
hidden_size = 768
batch_first = True

torch.set_default_device('cuda')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

idx2utt, utt2idx, idx2emo, emo2idx, idx2speaker,\
        speaker2idx, weight_matrix, my_dataset_train, my_dataset_test,\
        global_speaker_info, speaker_dialogues, speaker_emotions, \
        speaker_indices, utt_len, global_speaker_info_test, speaker_dialogues_test, \
        speaker_emotions_test, speaker_indices_test, utt_len_test = load_efr()
    
def get_train_test_loader(bs):
    print(len(my_dataset_train))
    train_data_iter = data.DataLoader(my_dataset_train, batch_size=bs)
    test_data_iter = data.DataLoader(my_dataset_test, batch_size=bs)
    
    return train_data_iter, test_data_iter
    
def train(model, train_data_loader, epochs):
    class_weights2 = torch.FloatTensor(weights2).to(device)
    criterion2 = nn.CrossEntropyLoss(weight=class_weights2,reduction='none').to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=5e-8,weight_decay=1e-5)
    
    max_f1_2 = 0
   
    for epoch in tqdm(range(epochs)):
        print("\n\n-------Epoch {}-------\n\n".format(epoch+1))
        model.train()
        
        avg_loss = 0
       
        y_true2 = []
        y_pred2 = []
            
        for i_batch, sample_batched in tqdm(enumerate(train_data_loader)):
            dialogue_ids = sample_batched[0].tolist()
            inputs = sample_batched[1].to(device)
            targets2 = sample_batched[3].to(device)
            
            # Creating the speaker_ids
            speaker_ids = []
            for d_ids_list in dialogue_ids:
              sp_id_list = [0] * len(d_ids_list)
              for ix, d_id in enumerate(d_ids_list):
                sp_id = global_speaker_info[d_id][0]
                sp_id_list[ix] = sp_id
              speaker_ids.append(sp_id_list)
            
            optimizer.zero_grad()
            
            _,outputs = model(inputs,dialogue_ids,speaker_ids,utt_len)
            
            loss = 0
            for b in range(outputs.size()[0]):
              loss2 = 0
              
              for s in range(utt_len[dialogue_ids[b][0]]):
                pred2 = outputs[b][s]
                pred_flip = torch.argmax(F.softmax(pred2.to(device),-1),-1)
                
                truth2 = targets2[b][s]

                y_pred2.append(pred_flip.item())
                y_true2.append(truth2.long().to(device).item())

                pred2_ = torch.unsqueeze(pred2,0)
                truth2_ = torch.unsqueeze(truth2,0)
                
                loss2 += criterion2(pred2_,truth2_)
              loss2 /= utt_len[dialogue_ids[b][0]]
            
            loss += loss2
            loss /= outputs.size()[0]
            avg_loss += loss

            loss.backward()            
            optimizer.step()
            
        avg_loss /= len(train_data_loader)
        
        print("Average Loss = ",avg_loss)
        if epoch%10==0:
            f1_2_cls,v_loss = validate(model, data_iter_test, epoch)
        
        # if f1_2_cls[1] > max_f1_2:
        #     print(f"Saving model at epoch {epoch}")
        #     max_f1_2 = f1_2_cls[1]
        #     torch.save(model.state_dict(), "./best_model.pth")

    return model

def validate(model, test_data_loader,epoch):
    print("\n\n***VALIDATION ({})***\n\n".format(epoch))
    
    class_weights2 = torch.FloatTensor(weights2).to(device)
    criterion2 = nn.CrossEntropyLoss(weight=class_weights2,reduction='none').to(device)

    model.eval()

    with torch.no_grad():
      avg_loss = 0
      y_true2 = []
      y_pred2 = []

      for i_batch, sample_batched in tqdm(enumerate(test_data_loader)):
            dialogue_ids = sample_batched[0].tolist()           
            inputs = sample_batched[1].to(device)
            targets2 = sample_batched[3].to(device)
            
            # Creating the speaker_ids
            speaker_ids = []
            for d_ids_list in dialogue_ids:
              sp_id_list = [0] * len(d_ids_list)
              for ix, d_id in enumerate(d_ids_list):
                sp_id = global_speaker_info[d_id][0]
                sp_id_list[ix] = sp_id
              speaker_ids.append(sp_id_list)
                       
            _,outputs = model(inputs,dialogue_ids,speaker_ids,utt_len)
            
            loss = 0
            for b in range(outputs.size()[0]):
              loss2 = 0
              
              for s in range(utt_len_test[dialogue_ids[b][0]]):
                pred2 = outputs[b][s]
                pred_flip = torch.argmax(F.softmax(pred2.to(device),-1),-1)
                
                truth2 = targets2[b][s]

                y_pred2.append(pred_flip.item())
                y_true2.append(truth2.long().to(device).item())

                pred2_ = torch.unsqueeze(pred2,0)
                truth2_ = torch.unsqueeze(truth2,0)
                
                loss2 += criterion2(pred2_,truth2_)
              loss2 /= utt_len_test[dialogue_ids[b][0]]
            
            loss += loss2
            loss /= outputs.size()[0]
            avg_loss += loss

      avg_loss /= len(test_data_loader)

      class_report = classification_report(y_true2,y_pred2)
      conf_mat2 = confusion_matrix(y_true2,y_pred2)

      print(class_report)
      print("Confusion Matrix: \n",conf_mat2)
    
      f1 = f1_score(y_true2,y_pred2)
      return f1,avg_loss

nclass = 2
utt_emsize = 768
personality_size = 100
nhid = 768
nlayers = 6
nhead = 2
dropout = 0.2
count_speakers = len(speaker2idx)
model = EFR_TX(weight_matrix, utt2idx, nclass, personality_size + utt_emsize, count_speakers, personality_size, nhead, nhid, nlayers, device, dropout).to(device)

weights2 = [1.0, 2.5]
data_iter_train, data_iter_test = get_train_test_loader(batch_size)
model = train(model, data_iter_train, epochs = 1000)