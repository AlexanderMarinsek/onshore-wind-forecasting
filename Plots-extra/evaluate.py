

def calc_mae(actual, forecast):

    length = len(actual)
    sum = 0

    for i in range(0, length):
        sum += abs( forecast[i] - actual[i] )

    return float(sum) / length


def calc_mape(actual, forecast):

    length = len(actual)
    sum = 0

    for i in range(0, length):
        if actual[i] != 0:
            error = abs( float( forecast[i] - actual[i] ) / actual[i] )
        else:
            error = 0

        sum+=error

    return float(sum) / length * 100


def calc_mse(actual, forecast):

    length = len(actual)
    sum = 0

    for i in range(0, length):
        sum += ( forecast[i] - actual[i] )**2

    return float(sum) / length


def calc_rmse(actual, forecast):

    length = len(actual)
    sum = 0

    for i in range(0, length):
        sum += ( forecast[i] - actual[i] )**2

    return (float(sum) / length)**(1/2)


def calc_errors(actual, forecast):

    if len(actual) != len(forecast):
        print ("Error in calculating errors!")
        quit ()

    return [
        calc_mae(actual, forecast),
        calc_mape(actual, forecast),
        calc_mse(actual, forecast),
        calc_rmse(actual, forecast)
    ]
