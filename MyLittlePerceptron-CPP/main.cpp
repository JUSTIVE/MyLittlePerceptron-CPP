#include "MultilayerPerceptron.h"

MultilayerPerceptron* mlp;
void preComputing() {
	mlp = new MultilayerPerceptron(3, 2);
	mlp->addHiddenLayer(3);

}

int main(int argc, char** argv) {
	mlp->init();
	mlp->resetWeights();




	delete mlp;
}

