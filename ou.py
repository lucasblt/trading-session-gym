import numpy as np
import matplotlib.pyplot as plt
import random

def _sim_price_dynamics(session_prices, OU_theta = 0.001, OU_mu = 0, OU_sigma = 1, dt = 1):
    #np.random.seed(133)

    for i in range(1, len(session_prices)):
        dW = np.random.randn()#rd.normal()
        delta_p = OU_theta * (session_prices[i-1] - session_prices[i]) * dt + 0.1 * dW
        session_prices[i] = session_prices[i] + delta_p

    dW = np.random.randn()  #rd.normal()
    delta_t = 0.01*OU_theta * (OU_mu - session_prices[0]) * dt + 0.3 * dW

    session_prices[0] += delta_t
    return session_prices

if __name__ == '__main__':
    #np.random.seed(122)
    init_price = 0
    session_prices = np.full(13, init_price, dtype='float32')
    prices = np.array(session_prices)

    for i in range(24*60*10):
        prices = np.vstack([prices, _sim_price_dynamics(session_prices, OU_mu = init_price)])


    print(prices)

    plt.figure()
    plt.plot(prices[:,:])
    plt.show()
