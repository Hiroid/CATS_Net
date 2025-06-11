import fasttext
import fasttext.util as ft_util

data_path = 'D:\Projects\Context Dependent Learning\datafile'
# ft_util.download_model('en', if_exists='ignore', datapath=datapath)
def word2vec(name_list, dimension, datapath=data_path):
    ft = fasttext.load_model(datapath +'\cc.en.300.bin')
    ft_util.reduce_model(ft, dimension)
    vector_list = []
    for name in name_list:
        vector_list.append(ft.get_word_vector(name))
    return vector_list
