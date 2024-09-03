# FlashAttention Learning Project

My personal project to self-learn and implement FlashAttention, based on studying [Tridao's FA repo](https://github.com/Dao-AILab/flash-attention).

## About
I read 2 repos [Tridao's FA repo](https://github.com/Dao-AILab/flash-attention), [66Ring's Tiny FA Implementation](https://github.com/66RING/tiny-flash-attention) and then write the simplified version with CuTe. Throughout this project, I've added detailed comments on everything I didn't initially understand. I hope these notes can help others who are also learning about FlashAttention.
## Build and Run
```
git submodule init
git submodule update
python setup.py install
python test.py
```
## Benchmark
```
--------------------------------------------------------------------------------
Test: b=4, h=32, s=1024, d=64
Naive Torch: 14.43ms
CuTe Flash:  1.02ms
Speedup: 14.09x
--------------------------------------------------------------------------------
Test: b=4, h=32, s=2048, d=32
Naive Torch: 83.29ms
CuTe Flash:  2.09ms
Speedup: 39.89x
--------------------------------------------------------------------------------
Test: b=4, h=32, s=2048, d=64
Naive Torch: 85.10ms
CuTe Flash:  3.31ms
Speedup: 25.73x
--------------------------------------------------------------------------------
```
Feel free to customize the config in test.py. 
Note that this implementation supports head dimensions of 32 and 64 only.

## Warning
As this is a learning project, some of my comments and interpretations might be incorrect. If you spot any errors or have suggestions for improvement, please feel free to open an issue.

## Credit
- [Tridao's FA repo](https://github.com/Dao-AILab/flash-attention)
- [66Ring's Tiny FA Implementation](https://github.com/66RING/tiny-flash-attention)
