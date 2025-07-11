import numpy as np
import cv2

class CelularAutomata2d:

    def calculate_rule(self,rule):
        delimiter = rule.find('/')
        if delimiter == -1:
            raise ValueError("Invalid rule format. Expected format is 'birth/survival'.")
        birth_rule = rule[:delimiter]
        survival_rule = rule[delimiter + 1:]
        # Convert the rules to a list of integers
        birth_rule = [int(b) for b in birth_rule if b.isdigit()]
        survival_rule = [int(s) for s in survival_rule if s.isdigit()]
        rule = [birth_rule,survival_rule]
        return rule

    def __init__(self,initial_matrix,size,rule,neighbourhood):
        self.rule=self.calculate_rule(rule)

        if initial_matrix is None:
            self.matrix=np.zeros((size[0],size[1]),dtype=np.bool)
        else:
            self.matrix=initial_matrix
        self.size=size
        self.neighbourhood=neighbourhood

    def step(self,num_steps=1):
        for _ in range(num_steps):
            new_matrix = np.zeros_like(self.matrix, dtype=np.bool)
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    # Get the neighbours based on the neighbourhood type
                    neighbours = self.get_neighbours(i, j)
                    # Apply the rule to determine the new state
                    new_matrix[i, j] = self.apply_rule(self.matrix[i][j],neighbours)
            self.matrix = new_matrix
    
    def get_neighbours(self, i, j):
        if self.neighbourhood == "von_neumann":
            return self.von_neumann_neighbours(i, j)
        elif self.neighbourhood == "moore":
            return self.moore_neighbours(i, j)
        else:
            raise ValueError("Unknown neighbourhood type: {}".format(self.neighbourhood))
        
    def von_neumann_neighbours(self, i, j):
        num_rows, num_cols = self.size
        neighbours = []
        # Vneighbourds
        top    = (i - 1) % num_rows
        bottom = (i + 1) % num_rows
        left   = (j - 1) % num_cols
        right  = (j + 1) % num_cols

        neighbours.append(self.matrix[top, j])
        neighbours.append(self.matrix[bottom, j])
        neighbours.append(self.matrix[i, left])
        neighbours.append(self.matrix[i, right])

        return sum(neighbours)
    
    def moore_neighbours(self, i, j):
        num_rows, num_cols = self.size
        neighbours = []
        # Moore neighbourhoods
        top    = (i - 1) % num_rows
        bottom = (i + 1) % num_rows
        left   = (j - 1) % num_cols
        right  = (j + 1) % num_cols

        neighbours.append(self.matrix[top, j])
        neighbours.append(self.matrix[bottom, j])
        neighbours.append(self.matrix[i, left])
        neighbours.append(self.matrix[i, right])
        
        neighbours.append(self.matrix[bottom, left])
        neighbours.append(self.matrix[bottom, right])
        neighbours.append(self.matrix[top, left])
        neighbours.append(self.matrix[top, right])

        return sum(neighbours)
    
    def apply_rule(self, cell_value,neighbours):
        return neighbours in self.rule[cell_value]
    
    def show(self):
        cv2.imshow("Cellular Automata", self.matrix.astype(np.uint8) * 255)
        cv2.waitKey(0)
        

    def convert_to_int(self):
        # Convert matrix to bytes
        flat = self.matrix.flatten()
        int_list = []

        # Process in chunks of 8 bits
        for n in range(0, len(flat) - 7, 8):
            new_value = 0
            for i in range(8):
                new_value = (new_value << 1) | int(flat[n + i])
            int_list.append(new_value)
        print(int_list)


    def randomize(self,probability=0.5):
        self.matrix = np.random.rand(*self.size) < probability
        return self.matrix
    
if __name__ == "__main__":
    iteraciones = input("iteraciones: ")
    ca = CelularAutomata2d(None, (500, 500), "23/2", "moore")
    ca.randomize(0.5)  # Randomly initialize the matrix with a probability of 0.5
    ca.matrix[2, 2] = True  # Set a cell to True
    for n in range(int(iteraciones)):
        ca.step()
        ca.show()