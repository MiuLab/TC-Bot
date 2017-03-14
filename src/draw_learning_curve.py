'''
Created on Nov 3, 2016

draw a learning curve

@author: xiul
'''

import argparse, json
import matplotlib.pyplot as plt


def read_performance_records(path):
    """ load the performance score (.json) file """
    
    data = json.load(open(path, 'rb'))
    for key in data['success_rate'].keys():
        if int(key) > -1:
            print("%s\t%s\t%s\t%s" % (key, data['success_rate'][key], data['ave_turns'][key], data['ave_reward'][key]))
            

def load_performance_file(path):
    """ load the performance score (.json) file """
    
    data = json.load(open(path, 'rb'))
    numbers = {'x': [], 'success_rate':[], 'ave_turns':[], 'ave_rewards':[]}
    keylist = [int(key) for key in data['success_rate'].keys()]
    keylist.sort()

    for key in keylist:
        if int(key) > -1:
            numbers['x'].append(int(key))
            numbers['success_rate'].append(data['success_rate'][str(key)])
            numbers['ave_turns'].append(data['ave_turns'][str(key)])
            numbers['ave_rewards'].append(data['ave_reward'][str(key)])
    return numbers

def draw_learning_curve(numbers):
    """ draw the learning curve """
    
    plt.xlabel('Simulation Epoch')
    plt.ylabel('Success Rate')
    plt.title('Learning Curve')
    plt.grid(True)

    plt.plot(numbers['x'], numbers['success_rate'], 'r', lw=1)
    plt.show()
            
    
            
def main(params):
    cmd = params['cmd']
    
    if cmd == 0:
        numbers = load_performance_file(params['result_file'])
        draw_learning_curve(numbers)
    elif cmd == 1:
        read_performance_records(params['result_file'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cmd', dest='cmd', type=int, default=1, help='cmd')
    
    parser.add_argument('--result_file', dest='result_file', type=str, default='./deep_dialog/checkpoints/rl_agent/11142016/noe2e/agt_9_performance_records.json', help='path to the result file')
    
    args = parser.parse_args()
    params = vars(args)
    print(json.dumps(params, indent=2))

    main(params)