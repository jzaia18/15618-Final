ifeq (, $(shell which jemalloc-config))
JEMALLOC =
else
JEMALLOC = -L$(JEMALLOCLD) -ljemalloc 
endif
# TODO get jemalloc working

CXX = g++
CXXFLAGS = -Wall -std=c++17 -pthread -I parlaylib/include/ -I. -Wno-unknown-pragmas $(JEMALLOC)
OPTFLAGS = -O3

PARLAYLIB = parlaylib/
boruvkas: boruvkas.cpp boruvkas.h | $(PARLAYLIB)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ boruvkas.cpp

$(PARLAYLIB):
	git clone https://github.com/cmuparlay/parlaylib.git

clean:
	/bin/rm -rf *~ *.o boruvkas
