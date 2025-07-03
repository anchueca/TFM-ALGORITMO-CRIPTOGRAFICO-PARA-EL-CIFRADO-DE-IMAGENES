import numpy as np
import cv2
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

kernel_code = """
__global__ void evolve(unsigned char *current, unsigned char *next, int rule, int size) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= size) return;

    int left  = current[(i - 1 + size) % size];
    int center = current[i];
    int right = current[(i + 1) % size];

    int pattern = (left << 2) | (center << 1) | right;
    next[i] = (rule >> pattern) & 1;
}
"""

mod = SourceModule(kernel_code)
evolve_gpu = mod.get_function("evolve")

class ElementalCelularAutomata:

    def __init__(self,initial_state,size,rule):
        self.rule=np.int32(rule)
        self.size=size
        self.num_step=0

        if initial_state is None:
            seed = int(time.time())
            self.randomize(initial_state)
        else:
            if type(initial_state) is int:
                self.randomize(initial_state)
            else:
                self.state=initial_state

        self.history = self.state.copy()
        # GPU memory
        self.dev_current = cuda.mem_alloc(self.state.nbytes)
        self.dev_next = cuda.mem_alloc(self.state.nbytes)
        cuda.memcpy_htod(self.dev_current, self.state)

    def step(self,num_steps=1):
        for _ in range(num_steps):
            new_state = np.empty_like(self.state, dtype=bool)
            for i,idx in enumerate(self.state):
                value=(self.state[i-1] << 2) | (idx << 1) | self.state[(i+1) % self.size]
                new_state[i]=(self.rule>>value) & 1
                    
            self.state = new_state
            #self.history = np.vstack((self.history, self.state))
            self.num_step+=1

    def step_cuda(self, num_steps=1):
        for _ in range(num_steps):
            evolve_gpu(
                self.dev_current,
                self.dev_next,
                self.rule,
                np.int32(self.size),
                block=(256, 1, 1),
                grid=((self.size + 255) // 256, 1)
            )
            cuda.memcpy_dtoh(self.state, self.dev_next)
            # swap buffers
            self.dev_current, self.dev_next = self.dev_next, self.dev_current
            self.num_step += 1

    def cleanup(self):
        self.dev_current.free()
        self.dev_next.free()

    
    def show(self):
        cv2.imshow("Cellular Automata", self.state.astype(np.uint8) * 255)
        cv2.waitKey(0)
        

    def convert_to_int(self):
        # Convert matrix to bytes
        int_list = []

        # Process in chunks of 8 bits
        for n in range(0, len(self.state) - 7, 8):
            new_value = 0
            for i in range(8):
                new_value = (new_value << 1) | int(self.state[n + i])
            int_list.append(new_value)
        return int_list


    def randomize(self,seed):
        np.random.seed(seed)
        self.state = np.random.randint(0, 2, self.size).astype(bool)
    
if __name__ == "__main__":
    iteraciones = input("iteraciones: ")
    ca = ElementalCelularAutomata(None, 100, 30)
    for n in range(int(iteraciones)):
        ca.step_cuda(5)
        ca.show()
        ca.convert_to_int()