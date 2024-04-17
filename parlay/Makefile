ifeq (, $(shell which jemalloc-config))
JEMALLOC =
else
JEMALLOC = -L$(JEMALLOCLD) -ljemalloc 
endif
# TODO get jemalloc working

APP_NAME=boruvkas
DEBUG_NAME=boruvkas_debug

OBJS=boruvkas.o

CXX = g++
CXXFLAGS = -Wall -std=c++17 -pthread -I parlaylib/include/ -I. -Wno-unknown-pragmas
DEBUGFLAGS = -ggdb
OPTFLAGS = -O3

PARLAYLIB = parlaylib/

all: $(APP_NAME) $(DEBUG_NAME)

debug: $(DEBUG_NAME)

$(DEBUG_NAME): $(PARLAYLIB) $(OBJS)
	$(CXX) $(DEBUGFLAGS) $(CXXFLAGS) -o $@ $(OBJS)

$(APP_NAME): $(PARLAYLIB) $(OBJS)
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cpp %.h $(EXTRA_HEADERS)
	$(CXX) $(CXXFLAGS) -ggdb -c $<

$(PARLAYLIB):
	git clone https://github.com/cmuparlay/parlaylib.git


clean:
	/bin/rm -rf *~ *.o $(APP_NAME) $(DEBUG_NAME)
