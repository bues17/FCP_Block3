import random

import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np


def diff_model(ics, t, _alpha, _beta, _gamma, _delta):
    '''
    Calculates the gradient for the Lotka-Volterra model
        Inputs:
                t: current time- not used here, but odeint expects to pass this argument so we must include it
                alpha: birth rate of the prey
                beta: effect of predator on preys growth rate
                gamma: effect of prey on predators growth rate
                delta: death rate of predator
                x: population density of prey.
                y: population density of predator.
        Outputs:
                the gradient of the model
     '''
    _x = ics[0]
    _y = ics[1]
    dxdt = _alpha * _x - _beta * _x * _y
    dydt = _delta * _x *  _y - _gamma * _y
    grad = [dxdt,dydt]
    return grad


def solve_model(t_lim, _alpha, _beta, _gamma, _delta, _ics):
    '''
    Solves a model using odeint. Does so for values of t between 0 and the one specified in the parameter
    '''
    t = np.linspace(0, t_lim)
    model = odeint(diff_model, _ics, t, (_alpha, _beta, _gamma, _delta,))
    return model, t


def plot_model(t, datalist):
    for i in range(datalist):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(t, data[:, 0], label='X(t)')
        ax1.set_title("Prey population over time")
        ax2 = fig.add_subplot(212)
        ax2.plot(t, data[:, 1], label='Y(t)')
        ax2.set_title("\nPredator population over time")
    plt.xlabel("time")
    plt.ylabel("population density")
    return fig


def run_sim(_alpha_list, _beta, _gamma, _delta, _ics):
    num_alphas = len(_alpha_list)
    datalist=[]
     # solve for 100 time steps
    t_max = 100
    for i in range(num_alphas-1):
        model,t = solve_model(t_max, _alpha_list[i], _beta, _gamma, _delta, _ics)
        datalist.append(model)

    fig = plot_model(t, datalist)
    fig.suptitle("Predator- prey population against time", fontsize=20)
    fig.tight_layout()
    plt.show()

alpha = 2/3
beta = 4/3
gamma = 1
delta = 1


# Create a tuple to represent the initial conditions
i_x = 10
i_y = 5
ics = (i_x, i_y)


run_sim(alpha,beta,gamma,delta,ics)


# argparse