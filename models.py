import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
import line_profiler
import utils


def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


def make_mlp(dim_list, activation="relu", batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class ActionEncoder(nn.Module):
    def __init__(
            self, obs_len, action_encoder_input_dim, action_encoder_hidden_dim,
    ):
        super(ActionEncoder, self).__init__()
        self.obs_len = obs_len
        self.action_encoder_input_dim = action_encoder_input_dim
        self.action_encoder_hidden_dim = action_encoder_hidden_dim

        self.actionEncoderLSTM = nn.LSTMCell(self.action_encoder_input_dim, self.action_encoder_hidden_dim)
        self.inputEmbedding = nn.Linear(2, self.action_encoder_input_dim)
        self.relu = nn.ReLU()

    def init_hidden(self, batch):
        return (
            torch.randn(batch, self.action_encoder_hidden_dim).cuda(),
            torch.randn(batch, self.action_encoder_hidden_dim).cuda(),
        )

    def forward(self, obs_traj):
        batch = obs_traj.shape[1]
        action_encoder_hidden_state, action_encoder_cell_state = self.init_hidden(batch)

        obs_traj_embedding = self.relu(self.inputEmbedding(obs_traj.contiguous().view(-1, 2)))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.action_encoder_input_dim)

        for i, input_data in enumerate(
            obs_traj_embedding[: self.obs_len].chunk(
                obs_traj_embedding[: self.obs_len].size(0), dim=0
            )
        ):
            action_encoder_hidden_state, action_encoder_cell_state = self.actionEncoderLSTM(
                input_data.squeeze(0), (action_encoder_hidden_state, action_encoder_cell_state)
            )

        return action_encoder_hidden_state


class GoalEncoder(nn.Module):
    def __init__(
            self, obs_len, goal_encoder_input_dim, goal_encoder_hidden_dim
    ):
        super(GoalEncoder, self).__init__()
        self.obs_len = obs_len
        self.goal_encoder_input_dim = goal_encoder_input_dim
        self.goal_encoder_hidden_dim = goal_encoder_hidden_dim

        self.goalEncoderLSTM = nn.LSTMCell(self.goal_encoder_input_dim, self.goal_encoder_hidden_dim)
        self.inputEmbedding = nn.Linear(2, self.goal_encoder_input_dim)
        self.relu = nn.ReLU()

        self.hidden2goal = nn.Linear(goal_encoder_hidden_dim, 2)
        self.flag = 0

    def init_hidden(self, batch):
        return (
            torch.randn(batch, self.goal_encoder_hidden_dim).cuda(),
            torch.randn(batch, self.goal_encoder_hidden_dim).cuda()
        )

    def forward(self, obs_goal):
        self.flag += 1

        batch = obs_goal.shape[1]
        goal_encoder_hidden_state, goal_encoder_cell_state = self.init_hidden(batch)

        obs_goal_embedding = self.relu(self.inputEmbedding(obs_goal.contiguous().view(-1, 2)))

        # if self.flag == 2:
        #     print(obs_goal_embedding)
        #     os._exit(0)

        obs_goal_embedding = obs_goal_embedding.view(-1, batch, self.goal_encoder_input_dim)

        for i, input_data in enumerate(
            obs_goal_embedding[: self.obs_len].chunk(
                obs_goal_embedding[: self.obs_len].size(0), dim=0
            )
        ):
            goal_encoder_hidden_state, goal_encoder_cell_state = self.goalEncoderLSTM(
                input_data.squeeze(0), (goal_encoder_hidden_state, goal_encoder_cell_state)
            )

        output = goal_encoder_hidden_state
        # output = self.hidden2goal(goal_encoder_hidden_state)
        return output


class GraphAttentionLayer(nn.Module):
    def __init__(self, n_head, f_in, f_out, distance_embedding_dim, attn_dropout, bias=True):
        super(GraphAttentionLayer, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.distance_embedding_dim = distance_embedding_dim

        self.w = nn.Parameter(torch.Tensor(n_head, 1, f_in, f_out))
        self.a = nn.Parameter(torch.Tensor(1, 1, f_out*2, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-2)

        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, h_self, h_other, goal, action):
        h_self_prime = torch.matmul(h_self, self.w)
        h_other_prime = torch.matmul(h_other, self.w)
        h_prime = torch.cat([h_self_prime, h_other_prime], dim=-1)

        h_prime = torch.matmul(h_prime, self.a)
        alpha = self.softmax(self.leaky_relu(h_prime))
        h_attn = torch.matmul(h_other, self.w)
        h_attn = alpha.repeat(1, 1, 1, self.f_out) * h_attn

        output = torch.mean(torch.sum(h_attn, dim=-2), dim=0)
        output = F.relu(output)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_head)
            + " -> "
            + str(self.f_in)
            + " -> "
            + str(self.f_out)
            + ")"
        )


class InteractionGate(nn.Module):
    def __init__(self, distance_embedding_dim, action_decoder_hidden_dim, ):
        super(InteractionGate, self).__init__()
        self.distance_embedding_dim = distance_embedding_dim
        self.action_decoder_hidden_dim = action_decoder_hidden_dim

        self.distEmbedding = nn.Linear(8, self.distance_embedding_dim)
        self.relu = nn.ReLU()
        self.gateFC = nn.Linear(self.distance_embedding_dim, self.action_decoder_hidden_dim)
        self.gateSigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # self.hiddenEmbedding = nn.Linear(self.action_decoder_hidden_dim, self.action_decoder_hidden_dim)
        # self.distEmbedding_2 = nn.Linear(self.action_decoder_hidden_dim * 3, self.distance_embedding_dim)
        # self.gate3_1 = nn.Linear(self.distance_embedding_dim * 4, 16)
        # self.gate3_2 = nn.Linear(16, 1)

    def cal_gate(self, goal, action):
        n = goal.size()[0]

        goal1 = goal.repeat(n, 1).view(n, n, -1)
        action1 = action.repeat(n, 1).view(n, n, -1)
        goal2 = goal1.transpose(1, 0)
        action2 = action1.transpose(1, 0)

        dist = torch.cat([action2, goal2, action1, goal1], dim=-1)
        gate = self.relu(self.distEmbedding(dist))
        gate = self.gateSigmoid(self.gateFC(gate))

        return gate

    def cal_gate_2(self, action_hidden_state, goal_hidden_state):
        n = action_hidden_state.size()[0]

        goal1 = goal_hidden_state.repeat(n, 1).view(n, n, -1)
        goal2 = goal1.transpose(1, 0)
        action2 = action_hidden_state.repeat(n, 1).view(n, n, -1).transpose(1, 0)

        hidden_state = torch.cat([goal1, goal2, action2], dim=-1)
        gate = self.relu(self.distEmbedding_2(hidden_state))
        gate = self.gateSigmoid(self.gateFC(gate))

        return gate

    def cal_gate_3(self, goal, action):
        n = goal.size()[0]
        gate = torch.ones([n, n+1, 1]).cuda()

        goal1 = self.relu(self.distEmbedding(goal.repeat(n, 1).view(n, n, -1)))
        action1 = self.relu(self.distEmbedding(action.repeat(n, 1).view(n, n, -1)))
        goal2 = self.relu(self.distEmbedding(goal.repeat(n, 1).view(n, n, -1).transpose(1, 0)))
        action2 = self.relu(self.distEmbedding(action.repeat(n, 1).view(n, n, -1).transpose(1, 0)))
        dist = torch.cat([action2, goal2, action1, goal1], dim=-1)

        dist = self.relu(self.gate3_1(dist))
        dist = self.gateSigmoid(self.gate3_2(dist))

    def forward(self, action_hidden_state, goal_hidden_state, goal, action,):
        num = goal.size()[0]
        gate = self.cal_gate(goal, action)

        temp_action_1 = action_hidden_state.repeat(num, 1).view(num, num, -1)
        gate_action = gate * self.tanh(temp_action_1)

        temp_goal = goal_hidden_state.repeat(num, 1).view(num, num, -1)

        mask_goal = torch.eye(num).cuda().unsqueeze(2).repeat([1, 1, self.action_decoder_hidden_dim])
        mask_action = torch.ones([num, num, self.action_decoder_hidden_dim]).cuda() - mask_goal

        goal_action = mask_goal * temp_goal + mask_action * gate_action

        return goal_action


class GAT(nn.Module):
    def __init__(
            self, distance_embedding_dim, action_decoder_hidden_dim,
            n_units, n_heads, dropout=0.2, alpha=0.2,
    ):
        super(GAT, self).__init__()
        self.n_heads = 4
        self.action_decoder_hidden_dim = action_decoder_hidden_dim

        self.interactionGate = InteractionGate(
            distance_embedding_dim=distance_embedding_dim, action_decoder_hidden_dim=action_decoder_hidden_dim
        )

        self.gatLayer = GraphAttentionLayer(
            n_head=self.n_heads, f_in=action_decoder_hidden_dim,
            f_out=action_decoder_hidden_dim, distance_embedding_dim=distance_embedding_dim, attn_dropout=dropout
        )

    def forward(self, action_hidden_state, goal_hidden_state, goal, action):
        n = action_hidden_state.size()[0]

        gated_h = self.interactionGate(
            action_hidden_state, goal_hidden_state, goal, action
        )

        h_self = action_hidden_state.repeat(n+1, 1).view(n+1, n, -1).transpose(1, 0)
        h_other = torch.cat([action_hidden_state.unsqueeze(1), gated_h], dim=1)

        output = self.gatLayer(h_self, h_other, goal, action)

        return output


class Attention(nn.Module):
    """
    交互模块, 刻画interaction
    input: hidden_state
    output: hidden_state'
    """
    def __init__(
            self, distance_embedding_dim, action_decoder_hidden_dim,
            n_units, n_heads, dropout, alpha
    ):
        super(Attention, self).__init__()
        self.gat = GAT(
            distance_embedding_dim, action_decoder_hidden_dim, n_units, n_heads, dropout, alpha
        )

    def forward(
            self, action_decoder_hidden_state, goal_decoder_hidden_state, goal, action, seq_start_end
    ):
        outputs = []
        for start, end in seq_start_end.data:
            curr_action_hidden_state = action_decoder_hidden_state[start:end, :]
            curr_goal_hidden_state = goal_decoder_hidden_state[start:end, :]
            curr_goal = goal[start:end, :]
            curr_action = action[start:end, :]
            curr_graph_embedding = self.gat(
                curr_action_hidden_state, curr_goal_hidden_state, curr_goal, curr_action
            )
            outputs.append(curr_graph_embedding)

        output_hidden_state = torch.cat(outputs, dim=0)
        return output_hidden_state


class Decoder(nn.Module):
    def __init__(
            self, obs_len, pred_len, n_units, n_heads, dropout, alpha, distance_embedding_dim,
            action_decoder_input_dim, action_decoder_hidden_dim, goal_decoder_input_dim, goal_decoder_hidden_dim
    ):
        super(Decoder, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.goal_decoder_input_dim = goal_decoder_input_dim
        self.goal_decoder_hidden_dim = goal_decoder_hidden_dim
        self.action_decoder_input_dim = action_decoder_input_dim
        self.action_decoder_hidden_dim = action_decoder_hidden_dim
        self.distance_embedding_dim = distance_embedding_dim

        self.goalDecoderLSTM = nn.LSTMCell(self.goal_decoder_input_dim, self.goal_decoder_hidden_dim)
        self.hidden2goal = nn.Linear(self.goal_decoder_hidden_dim, 2)
        self.goalEmbedding = nn.Linear(2, self.goal_decoder_input_dim)

        self.actionDecoderLSTM = nn.LSTMCell(self.action_decoder_input_dim, self.action_decoder_hidden_dim)
        self.actionEmbedding = nn.Linear(2, self.action_decoder_input_dim)
        self.hidden2action = nn.Linear(self.action_decoder_hidden_dim, 2)

        self.relu = nn.ReLU()

        self.attention = Attention(
            self.distance_embedding_dim, self.action_decoder_hidden_dim, n_units, n_heads, dropout, alpha
        )

        self.flag = 0

    def forward(
            self, goal_input_hidden_state, action_input_hidden_state,
            traj_real, action_real, teacher_forcing_ratio, seq_start_end, training_step
    ):
        self.flag += 1

        goal_decoder_hidden_state = goal_input_hidden_state
        goal_decoder_cell_state = torch.zeros_like(goal_decoder_hidden_state).cuda()
        action_decoder_hidden_state = action_input_hidden_state
        action_decoder_cell_state = torch.zeros_like(action_decoder_hidden_state).cuda()

        pred_seq = []
        action_output = action_real[self.obs_len - 1]
        traj_output = traj_real[self.obs_len - 1]

        batch = action_real.shape[1]
        action_real_embedding = self.relu(self.actionEmbedding(action_real.contiguous().view(-1, 2)))
        action_real_embedding = action_real_embedding.view(-1, batch, self.action_decoder_input_dim)
        traj_real_embedding = self.relu(self.goalEmbedding(traj_real.contiguous().view(-1, 2)))
        traj_real_embedding = traj_real_embedding.view(-1, batch, self.goal_decoder_input_dim)

        if self.training:
            for i in range(self.pred_len):
                if training_step == 1:
                    goal_input_data = traj_real_embedding[-self.pred_len + i]
                elif training_step == 2:
                    goal_input_data = traj_real_embedding[-self.pred_len + i]
                    if random.random() < teacher_forcing_ratio:
                        action_input_data = action_real_embedding[-self.pred_len + i]
                    else:
                        action_input_data = self.relu(self.actionEmbedding(action_output))
                elif training_step == 3:
                    goal_input_data = self.goalEmbedding(traj_output)
                    action_input_data = self.actionEmbedding(action_output)
                elif training_step == 4:
                    if random.random() < teacher_forcing_ratio:
                        goal_input_data = traj_real_embedding[-self.pred_len + i]
                    else:
                        goal_input_data = self.relu(self.goalEmbedding(traj_output))
                    if random.random() < teacher_forcing_ratio:
                        action_input_data = action_real_embedding[-self.pred_len + i]
                    else:
                        action_input_data = self.relu(self.actionEmbedding(action_output))

                goal_decoder_hidden_state, goal_decoder_cell_state = self.goalDecoderLSTM(
                    goal_input_data.squeeze(0), (goal_decoder_hidden_state, goal_decoder_cell_state)
                )
                goal_output = self.hidden2goal(goal_decoder_hidden_state)

                if training_step == 1:
                    pred_seq += [goal_output]
                    continue

                if i % 12 == 4:
                    action_decoder_hidden_state = self.attention(
                        action_decoder_hidden_state, goal_decoder_hidden_state,
                        goal_output, traj_output, seq_start_end
                    )

                action_decoder_hidden_state, action_decoder_cell_state = self.actionDecoderLSTM(
                    action_input_data.squeeze(0), (action_decoder_hidden_state, action_decoder_cell_state)
                )

                action_output = self.hidden2action(action_decoder_hidden_state)
                traj_output = traj_output + action_output

                if training_step == 2 or 3:
                    pred_seq += [action_output]
        else:
            for i in range(self.pred_len):
                goal_input_data = self.relu(self.goalEmbedding(traj_output))
                goal_decoder_hidden_state, goal_decoder_cell_state = self.goalDecoderLSTM(
                    goal_input_data.squeeze(0), (goal_decoder_hidden_state, goal_decoder_cell_state)
                )
                goal_output = self.hidden2goal(goal_decoder_hidden_state)
                # pred_goal += [goal_output]

                if i % 12 == 4:
                    action_decoder_hidden_state = self.attention(
                        action_decoder_hidden_state, goal_decoder_hidden_state,
                        goal_output, traj_output, seq_start_end
                    )

                action_input_data = self.relu(self.actionEmbedding(action_output))
                action_decoder_hidden_state, action_decoder_cell_state = self.actionDecoderLSTM(
                    action_input_data.squeeze(0), (action_decoder_hidden_state, action_decoder_cell_state)
                )

                action_output = self.hidden2action(action_decoder_hidden_state)
                traj_output = traj_output + action_output
                pred_seq += [action_output]

        # pred_goal_output = torch.stack(pred_goal)
        # pred_action_output = torch.stack(pred_action)
        pred_seq_output = torch.stack(pred_seq)
        return pred_seq_output


class TrajectoryPrediction(nn.Module):
    def __init__(
            self, obs_len, pred_len,
            action_input_dim, action_encoder_hidden_dim,
            goal_input_dim, goal_encoder_hidden_dim, distance_embedding_dim,
            n_units, n_heads, dropout, alpha,
            noise_dim=(8,), noise_type="gaussian",
    ):
        super(TrajectoryPrediction, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.action_input_dim = action_input_dim
        self.action_encoder_hidden_dim = action_encoder_hidden_dim
        self.goal_input_dim = goal_input_dim
        self.goal_encoder_hidden_dim = goal_encoder_hidden_dim
        self.noise_dim = noise_dim
        self.noise_type = noise_type

        self.goal_decoder_hidden_dim = self.goal_encoder_hidden_dim + self.noise_dim[0]
        self.action_decoder_hidden_dim = self.action_encoder_hidden_dim + self.noise_dim[0]

        self.actionEncoder = ActionEncoder(
            obs_len=self.obs_len,
            action_encoder_input_dim=self.action_input_dim,
            action_encoder_hidden_dim=self.action_encoder_hidden_dim,
        )
        self.goalEncoder = GoalEncoder(
            obs_len=self.obs_len,
            goal_encoder_input_dim=self.goal_input_dim,
            goal_encoder_hidden_dim=self.goal_encoder_hidden_dim,
        )
        self.Decoder = Decoder(
            obs_len=self.obs_len,
            pred_len=self.pred_len,
            goal_decoder_input_dim=self.goal_input_dim,
            goal_decoder_hidden_dim=self.goal_decoder_hidden_dim,
            action_decoder_input_dim=self.action_input_dim,
            action_decoder_hidden_dim=self.action_decoder_hidden_dim,
            distance_embedding_dim=distance_embedding_dim,
            n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha,
        )
        self.flag = 0

    def add_noise(self, _input, seq_start_end):
        noise_shape = (seq_start_end.size(0),) + self.noise_dim
        z_decoder = get_noise(noise_shape, self.noise_type)
        _list = []
        for idx, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            _vec = z_decoder[idx].view(1, -1)
            _to_cat = _vec.repeat(end-start, 1)
            _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
        decoder_h = torch.cat(_list, dim=0)
        return decoder_h

    def forward(
            self, input_traj, input_action, seq_start_end, teacher_forcing_ratio=0.5, training_step=3
    ):
        self.flag += 1
        goal_encoder_hidden_state = self.goalEncoder(input_traj)
        action_encoder_hidden_state = self.actionEncoder(input_action)

        goal_hidden_state_noise = self.add_noise(goal_encoder_hidden_state, seq_start_end)
        action_hidden_state_noise = self.add_noise(action_encoder_hidden_state, seq_start_end)

        pred_seq = self.Decoder(
            goal_hidden_state_noise, action_hidden_state_noise,
            input_traj, input_action, teacher_forcing_ratio, seq_start_end, training_step,
        )

        return pred_seq


