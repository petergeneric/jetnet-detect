CXX = g++
TARGET_BUILD=buildYolov2
TARGET_RUN=runYolov2

CXXFLAGS=-O3 -Wall -Wextra -g -std=c++11 -I. -I/usr/local/cuda/include -I/usr/include/x86_64-linux-gnu
CXXFLAGS += `pkg-config opencv --cflags`

LIBDIRS= -L/usr/local/cuda/lib64
CVLIBS += `pkg-config opencv --libs`
LDLIBS=$(LIBDIRS) -lm -lstdc++ $(CVLIBS) -lnvinfer_plugin -lnvinfer -lcuda -lcublas -lcurand -lcudart

all: $(TARGET_BUILD) $(TARGET_RUN)

$(TARGET_BUILD): $(TARGET_BUILD).o
$(TARGET_RUN): $(TARGET_RUN).o

.PHONY: clean

clean:
	rm -f *.o $(TARGET_BUILD) $(TARGET_RUN)
