# =========== define objective function for unsupervised competitive learning ==========

def bingo(y_true, y_pred):
    # y_true can be ignored
    # find the biggest 3 outputs
    threshold = np.partition(y_pred, -3)[-3]            # last 3 elements would be biggest
    loss = map( lambda y:
        if y >= threshold:
            (1 - y)                # if it is the winner, ideal value = 1.0
        else:
            (y)                    # if it is loser, ideal value = 0.0
        , y_pred)

    return np.array(loss)
