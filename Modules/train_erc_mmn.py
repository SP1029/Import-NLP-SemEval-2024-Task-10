import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from models import ERC_MMN
from pickle_loader import load_erc

from torch.utils.tensorboard import SummaryWriter
import os

batch_size = 64
seq_len = 15
seq2_len = seq_len
emb_size = 768
hidden_size = 768
batch_first = True

experiment_id = 'erc_mmn_masac_batch_64_1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter(os.path.join('../tensorboard_logs', experiment_id))

idx2utt, utt2idx, idx2emo, emo2idx, idx2speaker,\
    speaker2idx, weight_matrix, my_dataset_train, my_dataset_test,\
    final_speaker_info, final_speaker_dialogues, final_speaker_emotions,\
    final_speaker_indices, final_utt_len = load_erc('../Pickles/MaSaC')

weight_matrix = weight_matrix.to(device)
train_cnt = len(my_dataset_train)

def get_train_test_loader(bs):
    train_data_iter = data.DataLoader(my_dataset_train,batch_size=bs)
    test_data_iter = data.DataLoader(my_dataset_test,batch_size=bs,drop_last=True)
    
    return train_data_iter, test_data_iter

def train(model, train_data_loader, start_epoch, epochs):
    class_weights1 = torch.FloatTensor(weights1).to(device)
    criterion1 = nn.CrossEntropyLoss(weight=class_weights1,reduction='none').to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)    
    max_f1_1 = 0
    
    try:
    
      for epoch in range(start_epoch, start_epoch + epochs):
          # print("\n\n-------Epoch {}-------\n\n".format(epoch+1))
          model.train()
          
          avg_loss = 0
          
          y_true1 = []
          y_pred1 = []
              
          for i_batch, sample_batched in tqdm.tqdm(enumerate(train_data_loader), ncols=100, total=len(train_data_loader), desc=f'Epoch {epoch}'):
              dialogue_ids = sample_batched[0].tolist()            
              inputs = sample_batched[1].to(device)
              targets1 = sample_batched[2].to(device)
                  
              optimizer.zero_grad()
              
              _, outputs = model(dialogue_ids, final_speaker_info, final_speaker_dialogues, final_speaker_emotions, final_speaker_indices, inputs)
              
              loss = 0
              for b in range(outputs.size()[0]):
                loss1 = 0
                for s in range(final_utt_len[dialogue_ids[b]]):
                  pred1 = torch.unsqueeze(outputs[b][s],dim=0).to(device)
                  truth1 = torch.LongTensor([targets1[b][s].item()]).to(device)

                  pred_emo = torch.argmax(F.softmax(pred1,-1),-1)
                  
                  y_pred1.append(pred_emo.item())
                  y_true1.append(truth1.item())

                  loss1 += criterion1(pred1,truth1)

                loss1 /= final_utt_len[dialogue_ids[b]]
                loss += loss1

              loss /= batch_size
              avg_loss += loss

              loss.backward()            
              optimizer.step()
              
          avg_loss /= len(train_data_loader)
          print("Average Loss",avg_loss)
          writer.add_scalar('average_loss', avg_loss.cpu().detach().numpy()[0], epoch+1)
          f1_1,v_loss, c_matrix = validate(model,data_iter_test,epoch)
          writer.add_scalar("F1 Score",f1_1, epoch+1)
          
          if f1_1 > max_f1_1:
              print(f"Saving model at epoch {epoch}")
              max_f1_1 = f1_1
              save_model(model,
                '../Models/',
                experiment_id,
                batch_size,
                start_epoch + epoch,
                c_matrix)
    except Exception as e:
      # print('Error in train', e)
      raise e
    finally:
      # pass
      if f1_1 > max_f1_1:
        save_model(model,
          '../Models/',
          experiment_id,
          batch_size,
          start_epoch + epochs,
          c_matrix)

    return model

def save_model(model, model_dir, experiment_id, batch_size, epochs, c_matrix = None):
  """Save the model to disk

  Args:
      model (nn.Module): Model object. Model weights (state_dict()) 
                      and model object both are stored on disk
      model_dir (str): Path to save the model
      experiment_id (str): The ID of the experiment
                              `<some_id>_<exp_number>`
      batch_size (int): Batch size
      c_matrix (_type_, optional): confusion matrix. Defaults to None.
  """
  torch.save({
    'model_state_dict': model.state_dict(),
    'model': model,
    'confusion_matrix': c_matrix,
    'epochs': epochs,
    'hyperparameter':{
      'batch_size': batch_size,
      'seq_len': seq_len,
      'seq2_len': seq2_len,
      'emb_size': emb_size,
      'hidden_size': hidden_size,
      'batch_first': batch_first}
    }, os.path.join(model_dir, experiment_id + '.pkl'))
  
def load_model_if_exists(model_dir, experiment_id):  
  state = torch.load('../Models/MaSaC_ERC_MMN.pkl')
  model = state['model']
  model.load_state_dict(state['model_state_dict'])
  return model

def validate(model, test_data_loader,epoch):
    # print("\n\n***VALIDATION ({})***\n\n".format(epoch))
    class_weights1 = torch.FloatTensor(weights1).to(device)
    criterion1 = nn.CrossEntropyLoss(weight=class_weights1,reduction='none')
    
    model.eval()

    with torch.no_grad():
      avg_loss = 0
        
      y_true1 = []
      y_pred1 = []

      for i_batch, sample_batched in tqdm.tqdm(enumerate(test_data_loader), ncols=100, total=len(test_data_loader), desc=f'Validating {epoch}'):
            dialogue_ids = sample_batched[0].tolist()
            dialogue_ids = [train_cnt+d for d in dialogue_ids]
            inputs = sample_batched[1].to(device)
            targets1 = sample_batched[2].to(device)

            _, outputs = model(dialogue_ids, final_speaker_info, final_speaker_dialogues, final_speaker_emotions, final_speaker_indices, inputs, mode="valid")
            
            loss = 0
            for b in range(outputs.size()[0]):
              loss1 = 0
              for s in range(final_utt_len[dialogue_ids[b]]):
                pred1 = torch.unsqueeze(outputs[b][s],dim=0).to(device)
                truth1 = torch.LongTensor([targets1[b][s].item()]).to(device)

                pred_emo = torch.argmax(F.softmax(pred1,-1),-1)
                
                y_pred1.append(pred_emo.item())
                y_true1.append(truth1.item())

                loss1 += criterion1(pred1,truth1)

              loss1 /= final_utt_len[dialogue_ids[b]]
              loss += loss1

            loss /= batch_size
            avg_loss += loss

      avg_loss /= len(test_data_loader)

      class_report = classification_report(y_true1,y_pred1)
      conf_mat1 = confusion_matrix(y_true1,y_pred1)

      print(class_report)
      print("Confusion Matrix: \n",conf_mat1)
      
      acc = accuracy_score(y_true1, y_pred1)
      writer.add_scalar('Accuracy', acc, epoch+1)
      wtd_f1 = f1_score(y_true1,y_pred1,average="weighted")
      return wtd_f1, avg_loss, conf_mat1
    
n_emotions = 8
# model = ERC_MMN(hidden_size,weight_matrix,utt2idx,batch_size,seq_len, n_emotions=n_emotions).to(device)
model = load_model_if_exists(model_dir='../Models', experiment_id=experiment_id)
# weights1 = [1.0]*n_emotions
weights1 = [57.3859649122807,5.379934210526316,19.58682634730539,11.377391304347826,16.034313725490197,2.2101351351351353,11.558303886925795,17.7289972899729]
data_iter_train, data_iter_test = get_train_test_loader(batch_size)

model = train(model, data_iter_train, start_epoch = 100, epochs = 100)