import matplotlib.pyplot as plt

def plot(y_true, y_pred, horizon):
    """
        Input: 
            - horizon
            - y_pred
            - y_true
    """

    plt.figure(figsize=(12, 5))
    plt.plot(y_true.values, label="Truth data")
    plt.plot(y_pred, label="Prediction")

    plt.legend()
    plt.title("Weather forecast {}-day ahead".format(horizon))
    plt.grid()
    plt.savefig("./images/pred_{}.png".format(str(horizon)))
    plt.show()


