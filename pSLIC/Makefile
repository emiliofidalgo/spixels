PROGRAM = preemptive_slic

CXX        = g++
CFLAGS     = -O3 $(shell pkg-config --cflags opencv) 
LIBS      = $(shell pkg-config --libs opencv) 
SRC       = preemptiveSLIC.cpp demo_singleImage.cpp

$(PROGRAM): $(SRC) 
	$(CXX) $(CFLAGS) $(SRC) -o $(PROGRAM) $(LIBS)	
	
clean:
	@rm -rf $(PROGRAM)