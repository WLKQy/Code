class Enviroment():
    def __init__(self, W, num_ue, F, bn, dn, dist, f, pn, pi, action_space, state_shape, observation_space, it=0.5,
                 ie=0.5):
        self.W, self.num_ue, self.F = W, num_ue, F
        self.bn, self.dn, self.dist = bn, dn, dist
        self.f, self.it, self.ie = f, it, ie
        self.pn, self.pi = pn, pi
        self.action_space, self.state_shape = action_space, state_shape
        self.observation_space = observation_space
        # W = 10  # MHz 带宽
        # F = args.F  # Ghz/sec MEC 计算能力
        # f = 1  # Ghz/sec 本地 计算能力
        # num_ue = args.num_ue  # ue的个数
        #
        # dist = np.random.random(size=num_ue) * 200  # 每个ue的距离基站
        # bn = np.random.uniform(300, 500, size=num_ue)  # 输入量 kbits
        # dn = np.random.uniform(900, 1100, size=num_ue)  # 需要周期量 兆周期数 1Mhz = 1000khz = 1000 * 1000hz
        # #     tao = np.random.
        # it, ie = 0.5, 0.5  # 权重
        # pn, pi = 500, 100  # 传输功率， 闲置功率 mW

    # reset函数 得到智能体的观测值 任务量 cpu周期数 自身计算能力
    def reset(self):
        obs_n = np.zeros((self.num_ue, 4), dtype=np.float32)
        sjs = 0
        for i in range(self.num_ue):
            # 对于每个用户
            #             obs_n[i][1]= np.random.randint(900, 1100)#需要的cpu数量
            obs_n[i][0] = random.randint(15, 25)  # 任务数据量
            obs_n[i][1] = random.randint(25, 36)  # 车辆速度情况 80km/s
            obs_n[i][2] = random.randint(400, 500)  # 距离最左边的距离
            obs_n[i][3] = random.uniform(1.6, 3)  # 最大容忍时间 未定
        #             obs_n[i][4] = 0
        for q in range(self.num_ue):
            if (obs_n[i][0] * 20 * 1024) / (3.2 * 100 * 1000) > obs_n[i][3] and ((obs_n[i][0] * 1024) / (
                    0.5 * 6 * 10000 * np.log2(1 + (1.5 * 0.8 * pow(10, -4)) / (5.0 * pow(10, -14)))) + (
                                                                                         obs_n[i][0] * 1024 * 20) / (
                                                                                         F / 8 * 1000 * 1000)) > \
                    obs_n[i][3]:
                obs_n[i][3] = 100
        #         obs_n[0][2] = 487.5
        #         obs_n[2][2] = 485
        #         obs_n[3][2] = 486
        #         obs_n[4][2] = 483.5
        #         obs_n[1][2] = 484
        #         obs_n[0][2] = 2
        #         obs_n[1][2] = 2
        #         obs_n[2][2] = 1
        #         obs_n[3][2] = 2 #diyige
        #         obs_n[4][2] = 2
        #         obs_n[5][2] = 2

        #         obs_n[0][0] = 35
        #         obs_n[1][0] = 25
        #         obs_n[2][0] = 40
        #         obs_n[3][0] = 20
        #         obs_n[4][0] = 36
        #         obs_n[5][0] = 24

        #         obs_n[num_ue-1][3] = obs_n[num_ue-1][3] - 0.4
        #             sjs = random.random()
        #             if sjs <= 0.2:
        #                 obs_n[i][3] = (obs_n[i][1] / (obs_n[i][2] * 1000)) - 0.2
        #             elif sjs >= 0.8:
        #                 obs_n[i][3] = (obs_n[i][1] / (obs_n[i][2] * 1000)) + 0.2
        #             else:
        #                 obs_n[i][3] = obs_n[i][1] / (obs_n[i][2] * 1000)  # 任务容忍能力
        #         obs_n[0][3] = obs_n[2][3] - 0.2
        #         obs_n[num_ue -1] = obs_n[num_ue -1] + 0.2
        # 这里得到了n个用户的观测值  每个用户三个观测值 所以是 agent_num行 3列的数组
        return obs_n

    def get_state(self):
        state = np.zeros((self.num_ue, 3), dtype=np.float32)  # 创建空数组
        for i in range(self.num_ue):
            state[i][0] = obs_n[i][0]
            state[i][1] = obs_n[i][2]
            state[i][2] = 0  # done 默认为0 未完成
        # state[0][1] = 30
        # state[1][1] = 150
        # state[2][1] = 20
        # state[3][1] = 50
        # state[4][1] = 25
        #         state[5][1] = 50
        #         state[6][1] = 50
        return state  # 返回state状态矩阵\

    def get_statenew(self):
        new_state = np.zeros((self.num_ue, 3), dtype=np.int)
        for i in range(self.num_ue):
            new_state[i][0] = 0
            new_state[i][1] = new_obs_n[i][2]
            new_state[i][2] = 1
        return new_state

    def step(self, action_n):
        tt8 = np.zeros(num_ue, dtype=np.float32)
        tt9 = np.zeros(num_ue, dtype=np.float32)
        tcc = 0
        tt7 = 0
        sump = 0
        jg2 = 0
        for i in range(num_ue):
            if action_n[i] == 1:
                sump += 1
        k = 0
        for i in range(num_ue):
            if action_n[i] == 1:
                tcc += self.it * (obs_n[i][0] * 1024) / (
                            0.5 * 6 * 10000 * np.log2(1 + (1.5 * 0.8 * pow(10, -4)) / (5.0 * pow(10, -14))))
                tcc += self.ie * ((obs_n[i][0] * 1024) / (
                            0.5 * 6 * 10000 * np.log2(1 + (1.5 * 0.8 * pow(10, -4)) / (5.0 * pow(10, -14))))) * 1.5
                # 服务器计算
                tcc += self.it * (obs_n[i][0] * 1024 * 20) / (F / sump * 1000 * 1000)
                tcc += self.ie * ((obs_n[i][0] * 1024 * 20) / (F / sump * 1000 * 1000)) * 25
                tt7 += ((obs_n[i][0] * 1024) / (
                            0.5 * 6 * 10000 * np.log2(1 + (1.5 * 0.8 * pow(10, -4)) / (5.0 * pow(10, -14))))) * 1.5 + (
                                   (obs_n[i][0] * 1024 * 20) / (F / sump * 1000 * 1000)) * 25
                tt9[i] = (obs_n[i][0] * 1024) / (
                            0.5 * 6 * 10000 * np.log2(1 + (1.5 * 0.8 * pow(10, -4)) / (5.0 * pow(10, -14)))) + (
                                     obs_n[i][0] * 1024 * 20) / (F / sump * 1000 * 1000)
                tt8[i] = self.it * (obs_n[i][0] * 1024) / (0.5 * 6 * 10000 * np.log2(
                    1 + (1.5 * 0.8 * pow(10, -4)) / (5.0 * pow(10, -14)))) + self.ie * ((obs_n[i][0] * 1024) / (
                        0.5 * 6 * 10000 * np.log2(
                    1 + (1.5 * 0.8 * pow(10, -4)) / (5.0 * pow(10, -14))))) * 1.5 + self.it * (
                                 obs_n[i][0] * 1024 * 20) / (F / sump * 1000 * 1000) + self.ie * (
                                     (obs_n[i][0] * 1024 * 20) / (F / sump * 1000 * 1000)) * 25
            else:
                if k < num_kx and statell[k] > 3.2:
                    tcc += self.it * (obs_n[i][0] * 1024) / (
                                0.5 * 8 * 10000 * np.log2(1 + (1.5 * 1.3 * pow(10, -4)) / (5.0 * pow(10, -14))))
                    tcc += self.ie * ((obs_n[i][0] * 1024) / (
                                0.5 * 8 * 10000 * np.log2(1 + (1.5 * 1.3 * pow(10, -4)) / (5.0 * pow(10, -14))))) * 1.5
                    # 空闲车辆计算
                    tcc += self.it * (obs_n[i][0] * 1024 * 20) / (statell[k] * 100 * 1000)
                    tcc += self.ie * ((obs_n[i][0] * 1024 * 20) / (statell[k] * 100 * 1000)) * cgl
                    tt7 += ((obs_n[i][0] * 1024 * 20) / (statell[k] * 100 * 1000)) * cgl + ((obs_n[i][0] * 1024) / (
                                0.5 * 8 * 10000 * np.log2(1 + (1.5 * 1.3 * pow(10, -4)) / (5.0 * pow(10, -14))))) * 1.5
                    tt9[i] = (obs_n[i][0] * 1024) / (
                                0.5 * 8 * 10000 * np.log2(1 + (1.5 * 1.3 * pow(10, -4)) / (5.0 * pow(10, -14)))) + (
                                         obs_n[i][0] * 1024 * 20) / (statell[k] * 100 * 1000)
                    tt8[i] = self.it * (obs_n[i][0] * 1024) / (0.5 * 8 * 10000 * np.log2(
                        1 + (1.5 * 1.3 * pow(10, -4)) / (5.0 * pow(10, -14)))) + self.ie * ((obs_n[i][0] * 1024) / (
                                0.5 * 8 * 10000 * np.log2(
                            1 + (1.5 * 1.3 * pow(10, -4)) / (5.0 * pow(10, -14))))) * 1.5 + self.it * (
                                         obs_n[i][0] * 1024 * 20) / (statell[k] * 100 * 1000) + self.ie * (
                                         (obs_n[i][0] * 1024 * 20) / (statell[k] * 100 * 1000)) * cgl
                    k += 1
                    jg2 += 1
                else:
                    tcc += self.it * (obs_n[i][0] * 20 * 1024) / (3.2 * 100 * 1000)
                    tcc += self.ie * ((obs_n[i][0] * 20 * 1024) / (3.2 * 100 * 1000)) * cgl
                    tt7 += ((obs_n[i][0] * 20 * 1024) / (3.2 * 100 * 1000)) * cgl
                    tt9[i] = (obs_n[i][0] * 20 * 1024) / (3.2 * 100 * 1000)
                    tt8[i] = self.it * (obs_n[i][0] * 20 * 1024) / (3.2 * 100 * 1000) + self.ie * (
                                (obs_n[i][0] * 20 * 1024) / (3.2 * 100 * 1000)) * cgl

        # 现在开始求new_obs_n,创建一个新数组吧
        new_obs_n = np.zeros((self.num_ue, 4), dtype=np.float32)
        for i in range(self.num_ue):
            new_obs_n[i][0] = 0
            new_obs_n[i][1] = obs_n[i][1]
            new_obs_n[i][2] = obs_n[i][2] + (obs_n[i][0] * 1024 / (
                        0.5 * 6 * 10000 * np.log2(1 + (1.5 * 0.8 * pow(10, -4)) / (5.0 * pow(10, -14)))) +
                                             (obs_n[i][0] * 1024 * 20) / (F * 1000 * 1000)) * obs_n[i][1]
            new_obs_n[i][3] = obs_n[i][3]

        # 求done_n
        done_n = np.ones((self.num_ue), dtype=int)
        # 表示都完成了 每个智能体都完成任务了

        # 这里从新求reward
        rew_n = np.zeros((self.num_ue), dtype=float)
        for i in range(self.num_ue):
            rew_n[i] = (-1) * tt8[i]

        return new_obs_n, rew_n, done_n, tcc, jg2, tt9, tt8, tt7

