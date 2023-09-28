import os
import sys
import re
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, 'module/text_classification'))
sys.path.append(os.path.join(root, 'module/multi_task'))
sys.path.append(os.path.join(root, 'module/description_recognition'))
from module.text_classification.inference import TextClassifier

config_ = {
    'checkpoint_lst': [os.path.join(root, 'model/description_model/Appearance_PretrainedBert_1e-05_64_None.pt')],
    'use_bert': True,
    'embd_path': os.path.join(root, 'model/word_embedding/tencent_small'),
    'model_config_lst': [{
        'is_state': False,
        'model_name': 'bert',
        'pretrained_model_path': os.path.join(root, 'model/bert_chinese_wwm_ext_pytorch')
        # 'pretrained_model_path': '/share/作文批改/model/bert/chinese_wwm_ext_pytorch'
    }],
    'max_seq_len': 75,
    'need_mask': True
}
if __name__ == "__main__":
    sent_list = [
        '她有着大大的眼睛，长长的头发',
        '他戴着一副眼睛，看上去很斯文的样子。'
    ]
    pretrained_model_path = config_['model_config_lst'][0]['pretrained_model_path']
    model = TextClassifier(config_['embd_path'], config_['checkpoint_lst'], config_['model_config_lst'],
                           pretrained_model_path)
    max_seq_len = config_['max_seq_len'] if 'max_seq_len' in config_ else 80
    need_mask = config_['need_mask'] if 'need_mask' in config_ else False
    pred_list, proba_list = model.predict_all_mask(sent_list, max_seq_len=max_seq_len, max_batch_size=20,
                                                   need_mask=need_mask)
    pos_sent_list = [sent_list[i] for i in range(len(pred_list)) if pred_list[i] == 1]
    print(pred_list, proba_list)
    print("外貌描写：",len(pos_sent_list))
    for sent in pos_sent_list:
        print(sent)

