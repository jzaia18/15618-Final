APP_NAME=boruvkas
DEBUG_NAME=boruvkas_debug

OBJS=boruvkas.o

CXX = g++
CXXFLAGS = -Wall -std=c++17 -m64 -I. -Wno-unknown-pragmas
DEBUGFLAGS = -ggdb
OPTFLAGS = -O3

all: $(APP_NAME) $(DEBUG_NAME)

debug: $(DEBUG_NAME)

$(DEBUG_NAME): $(OBJS)
	$(CXX) $(DEBUGFLAGS) $(CXXFLAGS) -o $@ $(OBJS)

$(APP_NAME): $(OBJS)
	$(CXX) $(OPTFLAGS) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cpp %.h $(EXTRA_HEADERS)
	$(CXX) $(CXXFLAGS) -ggdb -c $<

clean:
	/bin/rm -rf *~ *.o $(APP_NAME) $(DEBUG_NAME)
