## this is a skew iou implentation  
## you can find the detailed decsription in this url.  
[Link1](https://github.com/HsLOL/C-2PythonModule)  

## By the way  
these `polyiou.cpp` `polyiou.h` `polyiou.i` `setup.py` are original files.  
these `polyiou.py` `polyiou_wrap.cxx` `_polyiou.cpython-36m-x86_64-linux-gnu.so` is generated.  
Where swig -c++ -python polyiou.i to generate polyiou_wrap.cxx and polyiou.py;  
And python setup.py build_ext --inplace to generate _polyiou.cpython-36m-x86_64-linux-gnu.so