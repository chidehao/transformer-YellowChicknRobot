import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer
try:
    from utils import make_trg_mask
except:
    from model.utils import make_trg_mask
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
class XiaoHJDataset(Dataset):
    def __init__(self,path):
        self.srcs = []
        self.trgs = []
        self.src_masks = []
        self.trg_masks = []
        with open (path,'r',encoding='utf-8') as f:
            line = f.readline()
            m_cnt = 0
            while line:
                line = line.strip().split(' ')
                if line[0] == 'E':
                    m_cnt = 0
                    try:
                        line = f.readline().strip()
                    except:
                        line = ""
                    continue
                else:
                    m_cnt +=1
                    if m_cnt == 1:
                        try:
                            src_dic = tokenizer(line[1].strip(),return_tensors='pt',padding='max_length')
                        except:
                            src_dic = tokenizer("", return_tensors='pt',padding='max_length')
                        self.srcs.append(src_dic['input_ids'])
                        self.src_masks.append(make_trg_mask(src_dic['attention_mask']))
                    elif m_cnt == 2:
                        try:
                            trg_dic = tokenizer(line[1].strip(),return_tensors='pt',padding='max_length')
                        except:
                            trg_dic = tokenizer("", return_tensors='pt',padding='max_length')
                        self.trgs.append(trg_dic['input_ids'])
                        self.trg_masks.append(make_trg_mask(trg_dic['input_ids']))
                    try:
                        line = f.readline().strip()
                    except:
                        line = ""
    def __getitem__(self, index):
        src = self.srcs[index]
        trg = self.trgs[index]
        src_mask = self.src_masks[index]
        trg_mask = self.trg_masks[index]
        return(src,trg,src_mask,trg_mask)

    def __len__(self):
        return len(self.srcs)

if __name__ == '__main__':
    test_path = r'C:\Users\YRJ\Desktop\transformerXL\src\xiaohuangji50w_nofenci.conv'
    dataset = XiaoHJDataset(test_path)
    train_loader = DataLoader(dataset,batch_size=2)
    for batch in train_loader:
        srcs,trgs,src_mask,trg_mask = batch
        srcs.to('cuda')
        trgs.to('cuda')
        src_mask.to('cuda')
        trg_mask.to('cuda')
        print(srcs)