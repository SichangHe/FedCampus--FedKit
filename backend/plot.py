import os

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import rcParams
from telemetry.models import EvaluateInsTelemetryData

rcParams["font.size"] = 26


x = []
y = []

xmax = 0
xmin = 0
ymax = 0
ymin = 0


def append(data_x, data_y):
    num = random.randint(1, 10)
    data_x.append(num)
    data_y.append(random.randint(1, 10))


fig, ax = plt.subplots()
ax.set_xlabel("Rounds")
ax.set_ylabel("Training Loss")

(line,) = ax.plot(x, y, linewidth=5, color="black")


# FuncAnimation动画回调函数
def update(i):
    y = EvaluateInsTelemetryData.objects.all().values_list("loss", flat=True)
    x = range(1, len(y) + 1)

    # 更新图表数据
    line.set_xdata(x)
    line.set_ydata(y)
    ax.set_xlim(0, 60)
    ax.set_ylim(min(y) - 100, max(y))

    # 重绘图表
    fig.canvas.draw()


ani = animation.FuncAnimation(fig, update, interval=60)
plt.title("FedCampus Real-Time Training Loss", fontsize=26)
plt.show()
# a = EvaluateInsTelemetryData.objects.all().values_list("loss", flat=True)
# print(len(a))
