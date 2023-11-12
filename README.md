# udacity-cs344-colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/depctg/udacity-cs344-colab)

Google Colab Notebooks for Udacity CS344 - Intro to Parallel Programming

Finish CS344 Projects online using Google's free GPU!

## Getting Start

- Click the ‘Open in Colab’ button.
- Select notebook for your homework.
- Run the first cell to setup environment.
- Finish code in CUDA cell and execute the cell to save the code. rerun following cells to recompile and plot your result.

## TODO
- [ ] Make scripts & notebooks for Final

## 2023/11/11 note

1. all `.ipynb` files: change `git+git` in `!pip install git+http://github.com/depctg/nvcc4jupyter.git` to `git+http`.
2. file `CMakeLists.txt`: repalce the previous one with `set(CUDA_NVCC_FLAGS "-gencode;arch=compute_75,code=sm_75;")` for the T4 GPU used in Colab now.
3. file `hw1.ipynb`,`hw2.ipynb`：add parameter `cmap='gray'` to `imshow()` calls in order to display the last three images correctly, otherwise they will be green.
4. file `HW1.cpp`,`HW2.cpp`: add these two head files to fix compile errors like "XXX was not declared in this scope" 
```cpp
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/imgproc/types_c.h>
```
5. when iterating over elements in filter, you'd better row++ first, then col++, outherwise the value of some pixels  in your output will have some tricky difference from the reference :(