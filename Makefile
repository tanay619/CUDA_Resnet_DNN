# Makefile for running the ResNet CIFAR-10 training and evaluation

# Define the name of the output file
OUTPUT_FILE = output.txt

# Define the command to run the Python script
PYTHON_CMD = python3 src/resnet_cifar.py

.PHONY: all cpu gpu clean

all: cpu gpu

cpu:
	@echo "Running on CPU..."
	$(PYTHON_CMD) F > $(OUTPUT_FILE)

gpu:
	@echo "Running on GPU..."
	$(PYTHON_CMD) T > $(OUTPUT_FILE)

clean:
	@echo "Cleaning up..."
	rm -f $(OUTPUT_FILE)
