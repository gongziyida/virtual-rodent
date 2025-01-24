from virtual_rodent.network.propri_enc import MLPEnc
import virtual_rodent.network._test_model_rnn as TestModel

env, propri_dim, action_dim = '_test_cheetah', 17, 6
emb_dim = 100 # propri_dim
def model_init_method():
    propri_enc = MLPEnc(propri_dim, emb_dim, hidden_dims=(300,))
    actor = TestModel.Actor(emb_dim, action_dim)
    critic = TestModel.Critic(emb_dim)
    return TestModel.TestModel(propri_enc, [propri_dim], actor, critic, action_dim) 
