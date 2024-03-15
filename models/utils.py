import uuid

import matplotlib.pyplot as plt


def plot_history(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig(f"history-{str(uuid.uuid1())}.png", bbox_inches="tight")
    plt.show()
