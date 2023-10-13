# 初始化环境
env = env = Enviroment(W=10, num_ue=num_ue, F=F, bn=np.random.uniform(300, 500, size=num_ue),
                       dn=np.random.uniform(900, 1100, size=num_ue),
                       dist=np.random.uniform(size=num_ue) * 200,
                       f=1, it=0.5, ie=0.5, pn=500, pi=100, action_space=2, observation_space=(num_ue, 4),
                       state_shape=(num_ue, 3))  # done mec remain  # 实例化环境
# 初始化QMIXAgent
QMIXAgent = QMIX(name="QMIXAgent",
                 obs_shape=env.observation_space,  # env 中记得添上 TODO
                 act_space=env.action_space,
                 agent_num=env.num_ue,
                 states_shape=env.state_shape,
                 parameters=model_parameters,
                 #                      log_path=log_path,
                 #                      model_path=model_path,
                 create_summary_writer=True)

# 重新开始的地方

# model_path = "/modelc/"  + \
# "QMIX"
# log_path = "/logs/"  + \
# "QMIX"

# if os.path.exists('modelc/'):
#     pass
# else:
#     os.mkdir('modelc/')
stall = np.zeros(num_kx,dtype= np.float32)
for i in range(num_kx):
    stall[i] = 3.5
# stall[2] = 4
# stall[0] = 3
# # stall[num_kx-1] = 4
# stall[5] = 4
statell = sorted(stall,reverse=True)

episode = 0
epoch = 0
train_step = 0
lis= []
lisxh= []

while episode < train_parameters["num_episodes"]:  # num_episodes = 100000次 #episode = 0从0开始
    t_start = time.time()
    obs_n = env.reset()  # list(agent_num,obs_shape)  num条数据 其中每个智能体的观测信息维度
    state = env.get_state()  # 得到环境状态
    episode_rewards = [0.0]  # 每一次训练的奖励
    #       agent_rewards = [[0.0] for _ in range(env.n)]#每个智能体的奖励
    action_n = QMIXAgent.action(obs_n, evaluation=False)  # 返回的是所有智能体value值最大的动作 的数组  是一个数组
#     action_n= [1,1,1,1,1,1,1,1,1,1]
    new_obs_n, rew_n, done_n,tccc,jg2,tt9,tt8,tt7 = env.step(action_n)  # 将这个value最大的动作数组传入step函数中得到新的 obs re 等信息
    s_next = env.get_statenew()  # 得到下一个状态
    padded_n = np.empty([num_ue])
    padded_n = padded_n.tolist()
    done = all(done_n)
    reward = sum(rew_n)
    #     print(tt9)
    #     print("动作为{}奖励为{}完成时间为{}".format(action_n,rew_n,tt9))
#     print("第{}个循环的奖励为{},动作为{}消耗为{}".format(episode, reward, action_n, tccc))
    QMIXAgent.experience(state, s_next, obs_n, action_n, rew_n, new_obs_n, done_n, padded_n)  # 添加到buffer中
    if episode > model_parameters["batch_size"]:
        if (episode - model_parameters["batch_size"]) % 200 == 0:
            for i in range(env.num_ue):  # 这里是取经验开始训练。这里是对于每个智能体的经验池来讲的  #TODO
                if len(QMIXAgent.replay_buffers_n[i]) > QMIXAgent.max_replay_buffer_len:
                    popo = QMIXAgent.update()  # 如果经验够了，就一直到算法最后了
                    QMIXAgent.train(popo)
    #         print("第{}个循环的奖励为{}".format(episode,reward))
    #         if episode % 200 == 0:
    #             QMIXAgent.save_model()
    episode += 1
#         print("------------------------------------------------")
