import random

import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import odeint
import numpy as np
import argparse

'''
READ ME
Can run through terminal or through debugger,
- When running through the terminal use --initial to change the two population numbers
with number of prey followed by number of predators e.g "--input 10 5" with 10 prey and 5
predators
- To change an initial value usew eg "--alpha" followed by the value. If you
don't change the values, default conditions will be used.
- should be able to take multiple alphas, but is unable to graph them unfortunately.
'''

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
    fig = plt.figure()
    outer = gridspec.GridSpec(2, len(datalist), wspace=0.2, hspace=0.2)

    for sim in range(0,len(datalist)):

    # code for 'subplots within subplots' taken from
    # https://stackoverflow.com/questions/34933905/adding-subplots-to-a-subplot

        data = datalist[sim]
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                 subplot_spec=outer[sim], wspace=0.1, hspace=0.1)
        ax1 = plt.Subplot(fig, inner[sim])
        ax1.plot(t, data[:, 0], label='X(t)')
        fig.add_subplot(ax1)
        ax2 = plt.Subplot(fig, inner[sim])
        ax2.plot(t, data[:, 1], label='Y(t)')
        fig.add_subplot(ax2)
    plt.xlabel("time")
    plt.ylabel("population density")
    # this code doesn't quite work as intended unfortunately...
    return fig


def run_sim(_alpha_list, _beta, _gamma, _delta, _ics):
    num_alphas = len(_alpha_list)
    datalist=[]
     # solve for 100 time steps
    t_max = 100
    t=[]
    for sim in range(0,num_alphas):
        model,t = solve_model(t_max, _alpha_list[sim], _beta, _gamma, _delta, _ics)
        datalist.append(model)

    fig = plot_model(t, datalist)
    fig.suptitle("Predator- prey population against time", fontsize=20)
    plt.show()

# Default Values if nothing is inputted in the terminal.

alpha = [2/3] # Alpha will be passed in as a list so that you can have multiple values for it.
beta = 4/3
gamma = 1
delta = 1


# tuple to represent the initial conditions.
i_x = 10 # Prey population density.
i_y = 5 # predator population density.

def parse_args(alpha, beta, gamma, delta, i_x, i_y):
    parser=argparse.ArgumentParser()
    parser.add_argument("--initial", nargs="+", type=int)
    parser.add_argument("--alpha", nargs=1, type=float , help="Values for alpha")
    parser.add_argument("--beta", type=float, nargs=1)
    parser.add_argument("--delta", type=float, nargs=1)
    parser.add_argument("--gamma", type=float, nargs=1)
    args=parser.parse_args()
    if args.initial:
        i_x=args.initial[0]
        i_y=args.initial[1]
    if args.alpha:
        alpha=args.alpha
    if args.beta:
        beta=args.beta[0]
    if args.delta:
        delta=args.delta[0]
    if args.gamma:
        gamma=args.gamma[0]
    pass

    return alpha, beta, gamma, delta, i_x, i_y


alpha, beta, gamma, delta, i_x, i_y=parse_args(alpha, beta, gamma, delta, i_x, i_y) # updating for any passed in values
ics=(i_x,i_y)
run_sim(alpha,beta,gamma,delta,ics)