# ===============
#     Model
# ===============
import tensorflow as tf
#from tensorflow.python.ops import rnn, rnn_cell

n_feature_quan = 14
n_feature_qual = 27
n_classes = 3 # 2 = scores, 3 = Win/lose/equal
input_quan = tf.placeholder('float', [None, n_feature_quan])
input_qual = tf.placeholder('int64', [None, n_feature_qual])
y = tf.placeholder('float', [None, n_classes])

# setting embedding space (league, season, stage, team, player)
embed_size = 4 ##
vocab_size_league = 26000 # number league from api
embed_sp_league = tf.Variable(tf.random_normal([vocab_size_league, embed_size], -1.0, 1.0))
vocab_size_season = 30 # just reserve year
embed_sp_season = tf.Variable(tf.random_normal([vocab_size_season, embed_size], -1.0, 1.0))
vocab_size_stage = 1000 # just reserve actually have only 38
embed_sp_stage = tf.Variable(tf.random_normal([vocab_size_stage, embed_size], -1.0, 1.0))
vocab_size_team = 300000 # for now, max 274581 from api_id
embed_sp_team = tf.Variable(tf.random_normal([vocab_size_team, embed_size], -1.0, 1.0))
vocab_size_player = 800000 # number of player in input as a api_id
embed_sp_player = tf.Variable(tf.random_normal([vocab_size_player, embed_size], -1.0, 1.0))

# ================
#  Win Lose Draw
# ================

def model_RNN_v0(data_quan, data_qual):
    # setting
    n_hidden = 4
    n_layer = 1
    n_classes = 3
    dropout_rate = 0.6
    keep_rate = 1.0 - dropout_rate # rate drop = 1 - keep_rate
    hd_layer_out = {'weights': tf.Variable(tf.random_normal([n_hidden, n_classes]))
                  ,'biases': tf.Variable(tf.random_normal([n_classes]))}
    # quantity input
    x_quan = data_quan
    # quality word look up
    x_league = tf.slice(data_qual, [0,0], [-1,1])
    #embed_x_league = tf.nn.embedding_lookup(embed_sp_league, x_league) # shape [None, 1, embed_size]
    x_season = tf.slice(data_qual, [0,1], [-1,1])
    #embed_x_season = tf.nn.embedding_lookup(embed_sp_season, x_season) # shape [None, 1, embed_size]
    x_stage = tf.slice(data_qual, [0,2], [-1,1])
    #embed_x_stage = tf.nn.embedding_lookup(embed_sp_stage, x_stage) # shape [None, 1, embed_size]
    x_team = tf.slice(data_qual, [0,3], [-1,2])
    #embed_x_team = tf.nn.embedding_lookup(embed_sp_team, x_team) # shape [None, 2, embed_size]
    x_player = tf.slice(data_qual, [0,5], [-1,22])
    #embed_x_player = tf.nn.embedding_lookup(embed_sp_player, x_player) # shape [None, 22, embed_size]
    # reshape to matrice-tensor to sequence of vector
    ##embed_x_league = tf.reshape(embed_x_league, [-1, embed_size])
    ##embed_x_season = tf.reshape(embed_x_season, [-1, embed_size])
    ##embed_x_stage = tf.reshape(embed_x_stage, [-1, embed_size])
    ##embed_x_team = tf.reshape(embed_x_team, [-1, 2*embed_size])
    ##embed_x_player = tf.reshape(embed_x_player, [-1, 22*embed_size])
    # combine every fucking inputs
    ##data = tf.concat([x_quan, embed_x_league, embed_x_season\
    ##                         , embed_x_stage, embed_x_team, embed_x_player], 1)
    # get size of element in vector
    data_qual = tf.cast(data_qual, tf.float32)
    data = tf.concat([x_quan, data_qual],1)
    data = tf.reshape(data, [-1, 14+(1*27)])
    # Generate a n_input-element sequence of inputs
    data = tf.split(data ,14+(1*27) ,1)
    # LSTM layer
    rnn_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=keep_rate)
    rnn_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden)
    rnn_cell_2 = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_2, output_keep_prob=keep_rate)
    rnn_cells = tf.nn.rnn_cell.MultiRNNCell([rnn_cell, rnn_cell_2])#,rnn_cell.BasicLSTMCell(n_hidden)])
    # generate prediction
    outputs, states = tf.nn.static_rnn(rnn_cells, data, dtype=tf.float32)
    # we only want the last output
    output = tf.matmul(outputs[-1], hd_layer_out['weights']) + hd_layer_out['biases']
    output = tf.nn.softmax(output)
    return output

def model_RNN_v1(data_quan, data_qual):
    # setting
    n_hidden = [32]
    n_layer = len(n_hidden)
    n_classes = 3
    droprate = 0.8
    hd_layer_out = {'weights': tf.Variable(tf.random_normal([n_hidden[n_layer-1], n_classes]))
                  ,'biases': tf.Variable(tf.random_normal([n_classes]))}
    # quantity input
    x_quan = data_quan
    # quality word look up
    x_league = tf.slice(data_qual, [0,0], [-1,1])
    embed_x_league = tf.nn.embedding_lookup(embed_sp_league, x_league) # shape [None, 1, embed_size]
    x_season = tf.slice(data_qual, [0,1], [-1,1])
    embed_x_season = tf.nn.embedding_lookup(embed_sp_season, x_season) # shape [None, 1, embed_size]
    x_stage = tf.slice(data_qual, [0,2], [-1,1])
    embed_x_stage = tf.nn.embedding_lookup(embed_sp_stage, x_stage) # shape [None, 1, embed_size]
    x_team = tf.slice(data_qual, [0,3], [-1,2])
    embed_x_team = tf.nn.embedding_lookup(embed_sp_team, x_team) # shape [None, 2, embed_size]
    x_player = tf.slice(data_qual, [0,5], [-1,22])
    embed_x_player = tf.nn.embedding_lookup(embed_sp_player, x_player) # shape [None, 22, embed_size]
    # reshape to matrice-tensor to sequence of vector
    embed_x_league = tf.reshape(embed_x_league, [-1, embed_size])
    embed_x_season = tf.reshape(embed_x_season, [-1, embed_size])
    embed_x_stage = tf.reshape(embed_x_stage, [-1, embed_size])
    embed_x_team = tf.reshape(embed_x_team, [-1, 2*embed_size])
    embed_x_player = tf.reshape(embed_x_player, [-1, 22*embed_size])
    # combine every fucking inputs
    data = tf.concat([x_quan, embed_x_league, embed_x_season\
                             , embed_x_stage, embed_x_team, embed_x_player], 1)
    # get size of element in vector
    data = tf.reshape(data, [-1, 14+(embed_size*27)])
    # Generate a n_input-element sequence of inputs
    data = tf.split(data ,14+(embed_size*27) ,1)
    # LSTM layer
    cell = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(size, use_peepholes=True)\
            , input_keep_prob = 1.0 - droprate) for size in n_hidden]
    cells = tf.nn.rnn_cell.MultiRNNCell(cell)
    #cell = tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=True)
    #cell = tf.nn.rnn_cell.DropoutWrapper(cell[0], input_keep_prob=1.0-droprate)
    # generate prediction
    outputs, states = tf.nn.static_rnn(cells, data, dtype=tf.float32)
    # we only want the last output
    output = tf.matmul(outputs[-1], hd_layer_out['weights']) + hd_layer_out['biases']
    output = tf.nn.softmax(output)
    return output