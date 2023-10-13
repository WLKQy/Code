class QMIX:
    def __init__(self,
                 name,
                 obs_shape,
                 act_space,
                 agent_num,
                 states_shape,
                 parameters,
                 #                  log_path,
                 #                  model_path,
                 create_summary_writer):
        self.name = name
        self.obs_shape = obs_shape  # [Box(10,), Box(10,), Box(10,)] #每个agent的观测信息 应该是3吧
        self.n_actions = act_space  # [Discrete(5), Discrete(5), Discrete(5)]#应该是2 原实验应该是三个智能体每个智能体有五个动作可以选择
        self.n_agents = agent_num
        self.states_shape = states_shape  # states_shape=30
        self.parameters = parameters
        #         self.log_path = log_path
        #         self.model_path = model_path
        self.input_shape1 = obs_shape[1]

        #         print(self.n_agents)
        #         print(obs_shape)
        #         print(self.n_actions)
        #         for i in range(self.n_agents):
        #             self.eval_rnn_n[i] = [RNN(input_shape=self.input_shape1,#.shape返回几行几列
        #                                rnn_hidden_dim=64, n_actions=self.n_agents)]
        # #             print('done')
        #         self.eval_rnn_n = [RNN(input_shape=self.obs_shape[i].shape[0],#.shape返回几行几列
        #                                rnn_hidden_dim=64, n_actions=self.n_actions[i].n)
        #                            for i in range(self.n_agents)]  # 每个agent选动作的网络
        self.eval_rnn_n = [RNN(input_shape=self.input_shape1,  # .shape返回几行几列
                               rnn_hidden_dim=64, n_actions=self.n_actions)
                           for i in range(self.n_agents)]  # 每个agent选动作的网络
        self.target_rnn_n = [RNN(input_shape=self.input_shape1,
                                 rnn_hidden_dim=64, n_actions=self.n_actions)
                             for i in range(self.n_agents)]

        self.eval_qmix_net = QMIXNet(self.n_agents, self.states_shape)  # QMIXNet(5，10)  +=（num_ue,num_ue+2)
        self.target_qmix_net = QMIXNet(self.n_agents, self.states_shape)  # 这里需要看一下state——shape是谁  看原Qmixnet的输入

        # -------------------所有的网络都在一起训练，所以只需要一个优化器，同时更新n+1个网络的参数（智能体网络+qmix网络）-----------------------
        self.all_parameters = []
        for i in range(self.n_agents):
            self.all_parameters += list(self.eval_rnn_n[i].parameters())
        self.all_parameters += list(self.eval_qmix_net.parameters())
        self.optimizers = torch.optim.RMSprop(self.all_parameters, lr=parameters["lr"])
        # ----------------------------------------------------------------------------------------------------------------------

        self.replay_buffers_n = []
        for i in range(self.n_agents):
            self.replay_buffers_n.append(ReplayBuffer(self.parameters["buffer_size"]))
        self.max_replay_buffer_len = parameters["max_replay_buffer_len"]
        self.replay_sample_index = None

        # 为每一个智能体构建可视化训练过程

    #         if create_summary_writer:
    #             current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #             print(current_time)
    #             self.summary_writer = []
    #             for i in range(self.n_agents):
    #                 train_log_dir = self.log_path + '/QMIX_Summary_' + current_time + "agent_" + str(i)
    #                 self.summary_writer.append(SummaryWriter(train_log_dir))
    #                 print('Init alg QMIX' + "agent_" + str(i))

    def action(self, obs_n, evaluation=False):  # 问题出现再eval——rnn_n网络输出维度中
        action_n = []
        #         print(obs_n)
        for i, obs in enumerate(obs_n):  # 对于这个列表中的每个智能体的观测信息
            obs = torch.as_tensor([obs], dtype=torch.float32)  # 转换成张量 每行 tensor([4.,2.])
            action_value = self.eval_rnn_n[i](obs).detach().cpu()  ##将智能体的观测信息传入eval_rnn_net()得到这个动作的价值value
            if (obs_n[i][0] * 20 * 1024) / (3.2 * 100 * 1000) > obs_n[i][3] and ((obs_n[i][0] * 1024) / (
                    0.5 * 6 * 10000 * np.log2(1 + (1.5 * 0.8 * pow(10, -4)) / (5.0 * pow(10, -14)))) + (
                                                                                         obs_n[i][0] * 1024 * 20) / (
                                                                                         F / 8 * 1000 * 1000)) > \
                    obs_n[i][3]:
                action = 1
                # 卸载到边缘服务器或者设备运行
            else:
                if np.random.randn() > self.parameters['epsilon']:  # 根据策略 相当于dpn中的策略选择动作
                    action = torch.max(action_value, 1)[1].item()  # 这个观测信息，会有很多动作，选择value值最大的动作输出
                #                 print(action)
                else:
                    action = np.random.randint(0, 2)

                # action = 0
            action_n.append(action)  # 将动作加入n_action数组中？是数组,里面是所有智能体最大value的动作值
        #             print(action)
        self.parameters['epsilon'] *= 0.999
        return action_n

    # 将信息放入经验池
    def experience(self, state, s_next, obs_n, act_n, rew_n, new_obs_n, done_n, padded_n):
        for i in range(self.n_agents):
            self.replay_buffers_n[i].add(state, s_next, obs_n[i], [act_n[i]], [rew_n[i]], new_obs_n[i],
                                         [float(done_n[i])], [padded_n[i]])

    def save_model(self):
        for i in range(self.n_agents):
            torch.save(self.eval_rnn_n[i].state_dict(),
                       "modelc/qmix1" + "/QMIX_eval_nets_agent_" + str(i) + ".pth")

    # 直接到结束的那个算法
    def update(self):
        # collect replay sample from all agents
        replay_sample_index = self.replay_buffers_n[0].make_index(self.parameters['batch_size'])  # 从buffer池中选取经验
        state_n = []
        s_next_n = []
        obs_n = []
        act_n = []
        obs_next_n = []
        rew_n = []
        done_n = []
        padded_n = []
        act_next_n = []

        for i in range(self.n_agents):  # 对于这几个智能体
            # replay_sample_index = idxes
            state, s_next, obs, act, rew, obs_next, done, padded = self.replay_buffers_n[i].sample_index(
                replay_sample_index)
            # obs_next = torch.as_tensor(obs_next, dtype=torch.float32)
            state_n.append(torch.tensor(state, dtype=torch.float32))
            s_next_n.append(torch.tensor(s_next, dtype=torch.float32))
            obs_n.append(torch.tensor(obs, dtype=torch.float32))
            obs_next_n.append(torch.tensor(obs_next, dtype=torch.float32))
            act_n.append(torch.tensor(act, dtype=torch.int64))
            done_n.append(torch.tensor(done, dtype=torch.float32))
            rew_n.append(torch.tensor(rew, dtype=torch.float32))
            padded_n.append(torch.tensor(padded, dtype=torch.float32))
            # 这里是分别放入不同的数组

        for i, obs_next in enumerate(obs_next_n):  # 对于几个智能体的下一状态的观测值
            target_mu = self.target_rnn_n[i](obs_next)  # 放入target网络中得到target-
            act_next_n.append(torch.tensor(target_mu, dtype=torch.int64))  # 这是加到act_next_n数组中了 通过target网络生成的

        popo = state_n, s_next_n, obs_n, act_n, rew_n, obs_next_n, done_n, act_next_n, padded_n

        return popo

    #         summaries = self.train((state_n, s_next_n, obs_n, act_n, rew_n, obs_next_n, done_n, act_next_n,padded_n))#得到最后的公式值了

    #         self.update_target_weights(tau=self.parameters["tau"])#更新网络参数

    #         for i in range(self.n_agents):
    #             for key in summaries.keys():
    #                 self.summary_writer[i].add_scalar(key, summaries[key], global_step=train_step)
    #             self.summary_writer[i].flush()

    # 更新网络参数
    def update_target_weights(self, tau=1):
        for eval_rnn, target_rnn in zip(self.eval_rnn_n, self.target_rnn_n):
            for eval_param, target_param in zip(eval_rnn.parameters(), target_rnn.parameters()):
                target_param.data.copy_(tau * eval_param + (1 - tau) * target_param)  # 直接将eval_param赋值给target_param
        for qmix_eval_param, qmix_target_param in zip(self.eval_qmix_net.parameters(),
                                                      self.target_qmix_net.parameters()):
            qmix_target_param.data.copy_(tau * qmix_eval_param + (1 - tau) * qmix_target_param)

        # 调用train函数 得到的是啥

    def train(self, memories):
        state_n, s_next_n, obs_n, act_n, rew_n, obs_next_n, done_n, act_next_n, padded_n = memories
        q_evals_n = []
        q_targets_n = []

        for i in range(self.n_agents):
            # 计算每个智能体的q_eval，最后合成为q_evals_n
            q_eval = self.eval_rnn_n[i](obs_n[i])
            q_eval = torch.gather(q_eval, 1, act_n[i])  # tensor(1024,1)
            q_evals_n.append(q_eval)  # list(3,1)=[tensor(1024,1),tensor(1024,1),tensor(1024,1)]

            # 计算每个智能体的q_target,最后合成为q_targets_n
            q_next = self.target_rnn_n[i](obs_next_n[i]).detach()
            q_next = q_next.max(1)[0]
            q_target = q_next.unsqueeze(1)  # tensor(1024,1)
            q_targets_n.append(q_target)

        q_evals_n = torch.cat(q_evals_n, dim=-1)
        q_targets_n = torch.cat(q_targets_n, dim=-1)
        q_total_eval = self.eval_qmix_net(q_evals_n, state_n[0]).view(self.parameters['batch_size'], 1)  # ???TODO
        q_total_target = self.target_qmix_net(q_targets_n, s_next_n[0])  # TODO s_next_n 第一行？？为啥
        # 注意eval_qmix_net和target_qmix_net输入时，state_n和s_next_n的维度

        #         reward_norm = (rew_n[0] - rew_n[0].mean()) / rew_n[0].std()#这个可能也需要

        targets = reward + self.parameters['gamma'] * q_total_target.view(self.parameters['batch_size'],
                                                                          1)  # 这里的reward是算法得到的
        # 得到的公式最后的那个而结果

        # ----------------------------计算一个loss,更新参数--------------------------------------------------------
        padded_n = torch.cat(padded_n, dim=-1)  # TODO
        padded_n = torch.unsqueeze(padded_n, dim=0)
        mask = 1 - padded_n.float()
        td_error = (q_total_eval - targets).unsqueeze(0)
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error
        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.optimizers.zero_grad()  # 优化器
        loss.backward()
        self.optimizers.step()
        summaries = dict([['LOSS/loss', loss]])
        print(summaries)
        #         print("summaries")
        #         print(loss.item())
        #         print(episode)
        with open("/kaggle/working/ttr.txt", "a") as f:
            dd = str(loss.item())
            f.write("{}\n".format(dd))
        self.update_target_weights(tau=self.parameters["tau"])  # 更新网络参数
        return summaries

