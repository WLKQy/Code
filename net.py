class QMIXNet(nn.Module):
    def __init__(self, n_agents, state_shape, mixing_embed_dim=64):  # embed_dim：嵌入层
        super(QMIXNet, self).__init__()
        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))  # state_dim=30 因该是3*10 agent*10 所以这里的state_shape = (3,3)
        self.embed_dim = mixing_embed_dim  # embed_dim=64

        self.hyper_w_1 = nn.Linear(self.state_dim,
                                   self.embed_dim * self.n_agents)  # input:state_dim(30),output:64*3=192
        self.hyper_w_2 = nn.Linear(self.state_dim, self.embed_dim)  # input:state_dim(30),output:embed_dim(64)

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)  # input:state_dim(30),output:embed_dim(64)

        # V(s) instead of a bias for the last layers
        self.hyper_b_2 = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.embed_dim, 1))  # input:state_dim(30),output:1

    def forward(self, agent_qs, states):  # agent_qs：(batch_size, n_agents)
        bs = agent_qs.shape[0]  # bs=batch_size
        states = states.reshape(-1, self.state_dim)  # states:(batch_size, state_dim)=(1024, 30)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # agent_qs:(batch_size, 1, n_agents)=(1024, 1, 3)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))  # w1:(batch_size, embed_dim * n_agents)=(1024, 192)
        b1 = self.hyper_b_1(states)  # b1:(batch_size, mbed_dim)=(1024, 64)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)  # w1:(batch_size, n_agents, embed_dim)=(1024, 3, 64)
        b1 = b1.view(-1, 1, self.embed_dim)  # b1:(batch_size, 1, embed_dim)=(1024, 1, 64)
        hidden = f.elu(torch.bmm(agent_qs, w1) + b1)  # hidden:(batch_size, 1, embed_dim)=(1024, 1, 64)

        # Second layer
        w_2 = torch.abs(self.hyper_w_2(states)).view(-1, self.embed_dim,
                                                     1)  # w_2:(batch_size, embed_dim, 1)=(1024, 64, 1)
        # State-dependent bias
        b2 = self.hyper_b_2(states).view(-1, 1, 1)  # b2:(batch_size, 1, 1)=(1024, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_2) + b2  # y:(batch_size, 1, 1)=(1024, 1, 1)
        # Reshape and return
        q_tot = y.view(bs, -1, 1)  # q_tot:(1024, 1, 1)
        return q_tot


class RNN(nn.Module):
    def __init__(self, input_shape, rnn_hidden_dim, n_actions):
        super(RNN, self).__init__()
        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)  # input:obs_shape(10), output:rnn_hidden_dim(64)
        self.rnn = nn.Linear(rnn_hidden_dim, rnn_hidden_dim)  # input:rnn_hidden_dim(64), output:rnn_hidden_dim(64)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)  # input:rnn_hidden_dim(64), output:n_actions(3)

    def forward(self, obs):  # obs:(batch_size, obs_shape)=(1024, 10)
        x = f.relu(self.fc1(obs))  # x:(batch_size, rnn_hidden_dim)=(1024, 64)
        h = f.relu(self.rnn(x))  # h:(batch_size, rnn_hidden_dim)=(1024,64)
        q = self.fc2(h)  # q:(batch_size, action_shape)=(1024, 5)
        return q
