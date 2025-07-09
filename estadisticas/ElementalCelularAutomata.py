import numpy as np
#import cv2
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import sys


mod = SourceModule("""
__global__ void evolve(unsigned char *current, unsigned char *next, int rule, int size) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= size) return;

    int left  = current[(i - 1 + size) % size];
    int center = current[i];
    int right = current[(i + 1) % size];

    int pattern = (left << 2) | (center << 1) | right;
    next[i] = (rule >> pattern) & 1;
}
""")
evolve_gpu = mod.get_function("evolve")

class ElementalCelularAutomata:

    def __init__(self,initial_state,size,rule,num_buffered=2):
        self.rule=np.int32(rule)
        self.size=size
        self.num_step=0
        self.num_buffered = num_buffered  # Number of device buffers to cycle through

        if initial_state is None:
            initial_state = np.random.randint(0, 2, size, dtype=np.uint8)

        # Use pinned memory for fast copy from device
        self.state = cuda.pagelocked_empty(shape=self.size, dtype=np.uint8)
        self.state[:] = initial_state
        #self.history = self.state.copy().reshape(1, -1)

        # Pre-allocate GPU buffers
        self.dev_buffers = [cuda.mem_alloc(self.state.nbytes) for _ in range(num_buffered)]
        for buf in self.dev_buffers:
            cuda.memset_d8(buf, 0, self.state.nbytes)

        # Copy initial state to first GPU buffer
        cuda.memcpy_htod(self.dev_buffers[0], self.state)
        self.current_index = 0

    def step(self,num_steps=1):
        for _ in range(num_steps):
            new_state = np.empty_like(self.state, dtype=bool)
            for i,idx in enumerate(self.state):
                value=(self.state[i-1] << 2) | (idx << 1) | self.state[(i+1) % self.size]
                new_state[i]=(self.rule>>value) & 1
                    
            self.state = new_state
            #self.history = np.vstack((self.history, self.state.copy()))
            self.num_step+=1

    def step_cuda(self, num_steps=1):
        for _ in range(num_steps):
            current_buf = self.dev_buffers[self.current_index]
            next_index = (self.current_index + 1) % self.num_buffered
            next_buf = self.dev_buffers[next_index]

            evolve_gpu(
                current_buf,
                next_buf,
                self.rule,
                np.int32(self.size),
                block=(256, 1, 1),
                grid=((self.size + 255) // 256, 1)
            )

            # Copy result to pinned host memory (fast)
            cuda.memcpy_dtoh(self.state, next_buf)

            self.current_index = next_index
            #self.history = np.vstack((self.history, self.state.copy()))
            self.num_step += 1

    def cleanup(self):
        for buf in self.dev_buffers:
            buf.free()

    
    #def show(self):
        #cv2.imshow("Cellular Automata", self.history.astype(np.uint8) * 255)
        #cv2.waitKey(0)
        

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
    def convert_to_bitstream(self):
            """Devuelve los bits actuales como un bytearray empaquetado (cada byte contiene 8 bits)."""
            bitstream = bytearray()
            for i in range(0, len(self.state), 8):
                byte = 0
                for j in range(8):
                    if i + j < len(self.state):
                        byte = (byte << 1) | int(self.state[i + j])
                    else:
                        byte <<= 1  # Relleno con 0 si la longitud no es múltiplo de 8
                bitstream.append(byte)
            return bitstream

def main():
    if len(sys.argv) < 3:
        print("Uso: python script.py <longitud> <salida.bin>")
        sys.exit(1)

    size = int(sys.argv[1])
    output_file = sys.argv[2]
    rule = 30

    automata = ElementalCelularAutomata(initial_state=None, size=size, rule=rule)


    steps = 10000
    for _ in range(steps):
        automata.step_cuda()

    bitstream = automata.convert_to_bitstream()

    with open(output_file, 'wb') as f:
        f.write(bitstream)

    print(f"Secuencia de {len(bitstream)*8} bits escrita en: {output_file}")

if __name__ == "__main__":
    main()