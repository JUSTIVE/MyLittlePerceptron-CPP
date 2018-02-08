#include<gl/glut.h>
#include<gl/GLU.h>
#include<gl/GL.h>
#define _USE_MATH_DEFINES
#include<math.h>
#include<iostream>
#include<vector>
#include<stack>
#include "MultilayerPerceptron.h"

using namespace std;

vector<MultilayerPerceptron::TrainingElement> trainingSet;
int window[2];
float errorRateBuffer;
MultilayerPerceptron* mlp;
bool training = false;
int training_iteration=0;
float signinfier(float x) {
	return (x > 0.0f )? 1 : 0;
}

void dataParser() {
	FILE* fp = NULL;
	fp = fopen("data.txt","r");
	try
	{
		while (!feof(fp)) {
			vector<float>in;
			vector<float>out;
			float time;
			float sign;

			fscanf(fp,"%f %f",&time,&sign);
			in.push_back(time);
			out.push_back(signinfier(sign));
			MultilayerPerceptron::TrainingElement te(in,out);
			trainingSet.push_back(te);
		}
		fclose(fp);
		mlp->setTrainingSet(trainingSet);
		cout << "data read complete" << endl;
	}
	catch (const std::exception&)
	{
		cout << "data read error"<<endl;
	}
}

void preComputing() {
	mlp = new MultilayerPerceptron(1, 1);
	mlp->addHiddenLayer(3);
	mlp->addHiddenLayer(5);
	mlp->addHiddenLayer(4);
	mlp->init();
	mlp->resetWeights();
	dataParser();
}

void render() {

	if (training) {
		training_iteration++;
		float err = mlp->train(0.2f);
	}

	//glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	//mainrender
	glColor3f(1, 0, 0);
	glBegin(GL_TRIANGLES); {
		glVertex2f(0.5, 0.5);
		glVertex2f(-0.5, 0.5);
		glVertex2f(0, -0.5);
	}
	glEnd();

	glutSwapBuffers();
}
void render2() {
	//glutSetWindow(window[1]);
	//draw errorRate
	//glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);
	//mainrender
	glColor3f(0, 1, 0);
	glBegin(GL_TRIANGLES); {
		glVertex2f(0.5, 0.5);
		glVertex2f(-0.5, 0.5);
		glVertex2f(0, -0.5);
	}
	glEnd();

	glutSwapBuffers();
}

void keyboard(unsigned char key,int x, int y ) {
	switch (key)
	{
	case '1':
		training = true;
	default:
		break;
	}
	glutPostRedisplay();
}

int main(int argc, char** argv) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE);
	glutInitWindowSize(500, 500);

	window[0] = glutCreateWindow("artificial neural net classifier test");
	glClearColor(0, 0, 0, 0);
	glutDisplayFunc(render);
	glutKeyboardFunc(keyboard);

	window[1] = glutCreateWindow("errorRate");
	glClearColor(0, 0, 0, 0);
	glutDisplayFunc(render2);

	preComputing();
	glutMainLoop();
	delete mlp;
}

