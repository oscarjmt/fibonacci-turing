import csv
import time
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

class TuringMachine:
    def __init__(self, tape, initial_state='q0', final_state='qf'):
        self.tape = tape
        self.head_position = 0
        self.current_state = initial_state
        self.final_state = final_state
        self.transitions = {}

    def add_transition(self, current_state, current_symbol, next_state, next_symbol, direction):
        if (current_state, current_symbol) in self.transitions:
            raise Exception("Duplicate transition: ({}, {})".format(current_state, current_symbol))
        self.transitions[(current_state, current_symbol)] = (next_state, next_symbol, direction)

    def load_transitions(self, file_path):
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            for row in reader:
                current_state = str(row[0])
                current_symbol = str(row[1])
                next_state = str(row[2])
                next_symbol = str(row[3])
                direction = str(row[4])
                self.add_transition(current_state, current_symbol, next_state, next_symbol, direction)
    
    def run(self):
        start = time.time()
        while self.current_state != self.final_state:
            current_symbol = self.tape[self.head_position]
            if (self.current_state, current_symbol) not in self.transitions:
                raise Exception("No transition found: ({}, {})".format(self.current_state, current_symbol))
            next_state, next_symbol, direction = self.transitions[(self.current_state, current_symbol)]
            self.tape[self.head_position] = next_symbol
            if direction == 'R':
                self.head_position += 1
                if self.head_position == len(self.tape):
                    self.tape.append('b')
            elif direction == 'L':
                self.head_position -= 1
                if self.head_position < 0:
                    self.tape.insert(0, 'b')
                    self.head_position = 0
            self.current_state = next_state
        return time.time() - start
    
    def create_graph(self):
        graph = nx.DiGraph()
        for transition in self.transitions:
            source_state, symbol = transition
            target_state, new_symbol, direction = self.transitions[transition]
            graph.add_edge(source_state, target_state, label=f"{symbol}, {new_symbol}, {direction}")
        
        pos = nx.spring_layout(graph, k=4, iterations=700)
        plt.figure(figsize=(35, 25))  # set the figure size
        nx.draw_networkx_nodes(graph, pos, node_size=2000)
        nx.draw_networkx_edges(graph, pos, width=1.5)
        nx.draw_networkx_labels(graph, pos, font_size=15, font_family="arial")
        nx.draw_networkx_edge_labels(graph, pos, font_size=15, font_family="arial")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    results = []
    times = []
    for i in range(1, 19):
        tape = ['1'] * i
        tm = TuringMachine(tape)
        tm.load_transitions('transitions.csv')
        delta = tm.run()
        result = tm.tape
        #print(result)
        #print(len([x for x in result if x == '1']))
        results.append(len([x for x in result if x == '1']))
        times.append(delta)
    df = pd.DataFrame({"results": results, "times": times})
    df.index = df.index + 1
    
    mask = df.times > 0
    x = df.index[mask]
    y = df.times[mask]

    # Fit an exponential trendline
    popt, pcov = np.polyfit(x, np.log(y), 1, cov=True)
    a = np.exp(popt[1])
    b = popt[0]
    
    y_pred = a*np.exp(b*x)
    score = r2_score(y, y_pred)
    
    # Plot the scatterplot and trendline
    plt.scatter(df.index, df.times, label='data')
    xs = np.linspace(0, 19, 100)
    plt.plot(xs, a*np.exp(b*xs), 'r-', label='exponential trendline')
    plt.annotate("r2 = {:.4f}".format(score), (1, 25))
    plt.legend()
    plt.show()
    
    tm.create_graph()
    
    
