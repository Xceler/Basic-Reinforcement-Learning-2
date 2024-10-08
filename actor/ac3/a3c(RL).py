import gym 
import torch as T 
import torch.multiprocessing as mp 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions import Categorical 

class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr = 1e-3, betas = (0.9,0.99), eps = 1e-8, weight_decay =0):
        super(SharedAdam, self).__init__(params, lr = lr, betas= betas, eps = eps, weight_decay = weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p] 
                state['step'] = 0 
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)
                state['exp_avg'].share_memory_() 
                state['exp_avg_sq'].share_memory_()
            

        
class ActorCritic(nn.Module):
    def __init__(self,input_dims, n_actions, gamma = 0.99):
        super(ActorCritic, self).__init__() 
        self.gamma = gamma 
        self.pi1 = nn.Linear(*input_dims, 128)
        self.v1 = nn.Linear(*input_dims, 128) 
        self.pi = nn.Linear(128, n_actions) 
        self.v = nn.Linear(128,1)
        self.rewards = []
        self.actions = []
        self.states = [] 
    
    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward) 
    
    def clear_memory(self):
        self.states = [] 
        self.actions = []
        self.rewards = [] 
    
    def forward(self, state):
        pi1 = F.relu(self.pi1(state))
        v1 = F.relu(self.v1(state))
        pi = self.pi(pi1)
        v = self.v(v1)
        return pi, v 
    

    def calc_R(self, done):
        states = T.tensor(self.states, dtype = T.float) 
        _, v =self.forward(states) 
        R = v[-1] * (1- int(done))
        batch_return = [] 
        for reward in self.rewards[::-1]:
            R = reward + self.gamma * R 
            batch_return.append(R) 
        
        batch_return.reverse() 
        batch_return = T.tensor(batch_return, dtype = T.float) 
        return batch_return 
    

    def calc_loss(self, done):
        states = T.tensor(self.states, dtype =T.float)
        actions= T.tensor(self.actions, dtype = T.float) 
        returns = self.calc_R(done) 
        pi, values = self.forward(states) 
        values = values.squeeze()
        critic_loss = (returns - values) ** 2
        probs = T.softmax(pi, dim= 1)
        dist = Categorical(probs) 
        log_probs =dist.log_prob(actions) 
        actor_loss = -log_probs * (returns - values) 
        total_loss = (critic_loss + actor_loss).mean() 
        return total_loss 
    
    def choose_action(self, observation):
        state = T.tensor([observation], dtype =T.float) 
        pi, v= self.forward(state) 
        probs = T.softmax(pi, dim = 1) 
        dist = Categorical(probs) 
        action = dist.sample().numpy()[0] 
        return action 
    

def worker_process(global_actor_critic, optimizer, input_dims, n_actions, gamma, global_ep_idx, env_id, n_games, t_max):
    local_actor_critic = ActorCritic(input_dims, n_actions, gamma) 
    env = gym.make(env_id) 
    t_step = 1 
    while global_ep_idx.value < n_games:
        done = False 
        observation = env.reset() 
        score = 0 
        local_actor_critic.clear_memory() 
        while not done:
            action = local_actor_critic.choose_action(observation) 
            observation_, reward, done, info = env.step(action) 
            score += reward 
            local_actor_critic.remember(observation, action, reward) 
            if t_step % t_max == 0 or done:
                loss = local_actor_critic.calc_loss(done) 
                optimizer.zero_grad() 
                loss.backward() 
                for local_param, global_param in zip(
                    local_actor_critic.parameters(),
                    global_actor_critic.parameters()):

                    if local_param.grad is not None:
                        global_param._grad = local_param.grad 
                optimizer.step() 
                local_actor_critic.load_state_dict(
                    global_actor_critic.state_dict()
                )
                local_actor_critic.clear_memory() 
            
            t_step += 1 
            observation = observation_ 
        with global_ep_idx.get_lock():
            global_ep_idx.value += 1 
        print('Episode:', global_ep_idx.value, 'Reward', score)



if __name__ == '__main__':
    lr = 1e-4 
    env_id = 'CartPole-v0'
    n_actions = 2
    input_dims = [4] 
    N_GAMES= 1000 
    T_MAX = 6 
    global_actor_critic = ActorCritic(input_dims, n_actions) 
    global_actor_critic.share_memory() 
    optim  = SharedAdam(global_actor_critic.parameters(), lr = lr , betas = (0.92, 0.999))
    global_ep = mp.Value('i', 0) 

    with mp.Manager() as manager:
        processes = [ 
            mp.Process(target =worker_process, args=(
                global_actor_critic, optim, input_dims, n_actions, 0.99, global_ep, env_id, N_GAMES, T_MAX
            ))

            for _ in range(mp.cpu_count())
        ]

        for p in processes:
            p.start() 
        
        for p in processes:
            p.join() 