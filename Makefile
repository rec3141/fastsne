# fastsne build
#
#   make cpu                 fastsne_cpu: launcher + AVX-512 (skylake-avx512) / AVX2 (haswell) / generic builds
#                            (portable: a container built on one machine runs on any x86-64; the launcher picks
#                            the best build for the host)
#   make gpu                 fastsne: CUDA, one binary for V100 (sm_70), A30/A100 (sm_80), L40S/RTX 40 (sm_89)
#                            plus PTX for newer GPUs; set CUDA_ARCHS to change, NVCC to pick a toolkit,
#                            CUDA_INC / CUDA_LIB for extra -I / -L, -Xlinker -rpath flags
#   make native              fastsne_cpu tuned for this machine only (-march=native)
#   make install PREFIX=/usr/local   installs whatever was built, plus the viewer and tools under share/fastsne
#   STATIC=1                 self-contained binaries (the release builds): fastsne_cpu fully static (FFTW and
#                            OpenMP included), fastsne with cuFFT / cuBLAS / CUDA runtime linked in, so it
#                            needs only the NVIDIA driver
#
# CPU needs g++ with OpenMP and FFTW3 single precision (libfftw3-dev / conda fftw); GPU needs nvcc + cuFFT/cuBLAS.

CXX ?= g++
NVCC ?= nvcc
PREFIX ?= /usr/local
CXXFLAGS ?= -O3
CPU_FLAGS = -fopenmp -std=c++17 -Wno-psabi
CPU_LIBS = -lfftw3f_omp -lfftw3f -lm
FFTW_INC ?=
FFTW_LIB ?=
CUDA_INC ?=
CUDA_LIB ?=
CUDA_ARCHS ?= 70 80 89
ifeq ($(STATIC),1)
CPU_LDFLAGS = -static
CUDA_LIBS = -lcufft_static -lcublas_static -lcublasLt_static -lculibos -Xcompiler -static-libstdc++,-static-libgcc
else
CPU_LDFLAGS =
CUDA_LIBS = -lcufft -lcublas
endif
GENCODE = $(foreach a,$(CUDA_ARCHS),-gencode arch=compute_$(a),code=sm_$(a)) \
          -gencode arch=compute_$(lastword $(CUDA_ARCHS)),code=compute_$(lastword $(CUDA_ARCHS))

CPU_VARIANTS = fastsne_cpu.avx512 fastsne_cpu.avx2 fastsne_cpu.generic

.PHONY: all cpu gpu native install clean
all: cpu

cpu: $(CPU_VARIANTS) fastsne_cpu

fastsne_cpu.avx512: fastsne_cpu.cpp common.h
	$(CXX) $(CXXFLAGS) -march=skylake-avx512 -mtune=generic $(CPU_FLAGS) $(FFTW_INC) $(CPU_LDFLAGS) -o $@ $< $(FFTW_LIB) $(CPU_LIBS)
fastsne_cpu.avx2: fastsne_cpu.cpp common.h
	$(CXX) $(CXXFLAGS) -march=haswell -mtune=generic $(CPU_FLAGS) $(FFTW_INC) $(CPU_LDFLAGS) -o $@ $< $(FFTW_LIB) $(CPU_LIBS)
fastsne_cpu.generic: fastsne_cpu.cpp common.h
	$(CXX) $(CXXFLAGS) -march=x86-64 -mtune=generic $(CPU_FLAGS) $(FFTW_INC) $(CPU_LDFLAGS) -o $@ $< $(FFTW_LIB) $(CPU_LIBS)

fastsne_cpu: tools/fastsne_cpu.sh
	cp $< $@ && chmod +x $@

native: fastsne_cpu.cpp common.h
	$(CXX) $(CXXFLAGS) -march=native $(CPU_FLAGS) $(FFTW_INC) $(CPU_LDFLAGS) -o fastsne_cpu $< $(FFTW_LIB) $(CPU_LIBS)

gpu: fastsne
fastsne: fastsne.cu common.h
	$(NVCC) -O3 -std=c++14 $(GENCODE) $(CUDA_INC) -o $@ $< $(CUDA_LIB) $(CUDA_LIBS)

install:
	install -d $(PREFIX)/bin $(PREFIX)/share/fastsne/tools
	for f in fastsne fastsne_cpu $(CPU_VARIANTS); do [ -e $$f ] && install -m 755 $$f $(PREFIX)/bin/ || true; done
	install -m 644 viewer.html $(PREFIX)/share/fastsne/
	install -m 755 render.py tools/*.py $(PREFIX)/share/fastsne/tools/
	install -m 644 LICENSE $(PREFIX)/share/fastsne/

clean:
	rm -f fastsne fastsne_cpu $(CPU_VARIANTS)
