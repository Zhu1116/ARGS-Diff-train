import torch
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, ch_in, ch_out, num_hidden=None, time_embedding_dim=20, num_embeddings=1000):
        super(FCN, self).__init__()
        if num_hidden is None:
            num_hidden = [128, 256, 256, 128]
        model = []
        model.append(nn.Linear(ch_in + time_embedding_dim, num_hidden[0], bias=True))
        model.append(nn.ReLU6())

        for i in range(len(num_hidden) - 1):
            model.append(nn.Linear(num_hidden[i], num_hidden[i + 1], bias=True))
            model.append(nn.ReLU())

        model.append(nn.Linear(num_hidden[len(num_hidden) - 1], ch_out))

        self.model = nn.Sequential(*model)

        self.time_embedding = nn.Embedding(num_embeddings, time_embedding_dim)

    def forward(self, x, t):
        embedded_time = self.time_embedding(t[:, None])
        embedded_time = torch.squeeze(embedded_time, 1)
        x = torch.cat((x, embedded_time), dim=1)
        # embedded_time = embedded_time.expand(embedded_time.shape[0], x.shape[1], embedded_time.shape[2])
        # x = torch.cat((x, embedded_time), dim=2)
        x = self.model(x)
        return x

if __name__ == '__main__':
    x = torch.randn(64, 31)

    # 生成形状为(64, 1)的随机整数张量，数值范围在1到1000之间
    t = torch.randint(0, 999, (64,), dtype=torch.int)

    model = FCN(31, 31, [128, 256, 256, 128], time_embedding_dim=20)
    out = model(x, t)
    print(out.shape)














