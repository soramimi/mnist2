
#include "mnist.h"
#include "rwfile.h"
#include <functional>
#include <random>

class TwoLayerNet {
public:
	Layer layer1;
	Layer layer2;

	TwoLayerNet()
	{
		int input = 28 * 28;
		int hidden = 50;
		int output = 10;
		layer1.reset(input, hidden);
		layer2.reset(hidden, output);
	}

	Matrix predict(Matrix const &x)
	{
		return x.dot(layer1.weight).add(layer1.bias).sigmoid()
				.dot(layer2.weight).add(layer2.bias).softmax();
	}

	Matrix::real_t accuracy(Matrix const &x, Matrix const &t)
	{
		auto argmax = [](Matrix const &a, int row){
			int i = 0;
			for (size_t j = 1; j < a.cols(); j++) {
				if (a.at(row, j) > a.at(row, i)) {
					i = j;
				}
			}
			return i;
		};

		int rows = std::min(x.rows(), t.rows());
		Matrix y = predict(x);
		int acc = 0;
		for (int row = 0; row < rows; row++) {
			auto a = argmax(y, row);
			auto b = argmax(t, row);
			if (a == b) {
				acc++;
			}
		}
		return (Matrix::real_t)acc / rows;
	}

	void gradient(Matrix const &x, Matrix const &t, std::map<std::string, Matrix> *out)
	{
		out->clear();
		int batch_num = x.rows();

		Matrix a1 = x.dot(layer1.weight).add(layer1.bias);
		Matrix z1 = a1.sigmoid();
		Matrix a2 = z1.dot(layer2.weight).add(layer2.bias);
		Matrix y = a2.softmax();

		Matrix dy = y.sub(t).div(batch_num);

		(*out)["w2"] = z1.transpose().dot(dy);
		(*out)["b2"] = dy.sum();

		Matrix w2t = layer2.weight.transpose();
		Matrix dz1 = dy.dot(w2t);
		Matrix da1 = a1.sigmoid_grad().mul(dz1);

		(*out)["w1"] = x.transpose().dot(da1);
		(*out)["b1"] = da1.sum();
	}
};

int main()
{
	mnist::DataSet train;
	if (!train.load("train-labels-idx1-ubyte", "train-images-idx3-ubyte")) {
		fprintf(stderr, "failed to load mnist images and labels\n");
		exit(1);
	}

	mnist::DataSet t10k;
	if (!t10k.load("t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte")) {
		fprintf(stderr, "failed to load mnist images and labels\n");
		exit(1);
	}

	int iteration = 10000;
	int batch_size = 100;
	Matrix::real_t learning_rate = 0.1;

	std::random_device seed_gen;
	std::default_random_engine engine(seed_gen());
	std::normal_distribution<Matrix::real_t> dist(0.0, 0.1);
	auto Initialize = [&](Matrix *a){
		auto Rand = [&](){
			return dist(engine);
		};
		for (size_t i = 0; i < a->size(); i++) {
			a->data()[i] = Rand();
		}
	};

	TwoLayerNet net;
	Initialize(&net.layer2.weight);
	Initialize(&net.layer1.weight);

	unsigned int k = 0;
	for (int i = 0; i < iteration; i++) {
		Matrix x_batch;
		Matrix t_batch;
		for (int j = 0; j < batch_size; j++) {
			Matrix x, t;
			k = (k + rand()) % train.size();
			train.image_to_matrix(k, &x);
			train.label_to_matrix(k, &t);
			x_batch.add_rows(x);
			t_batch.add_rows(t);
		}

		{
			std::map<std::string, Matrix> grads;
			net.gradient(x_batch, t_batch, &grads);

			Matrix w1 = grads["w1"];
			Matrix b1 = grads["b1"];
			Matrix w2 = grads["w2"];
			Matrix b2 = grads["b2"];
			w1 = w1.mul(learning_rate);
			b1 = b1.mul(learning_rate);
			w2 = w2.mul(learning_rate);
			b2 = b2.mul(learning_rate);
			net.layer1.weight = net.layer1.weight.sub(w1);
			net.layer1.bias = net.layer1.bias.sub(b1);
			net.layer2.weight = net.layer2.weight.sub(w2);
			net.layer2.bias = net.layer2.bias.sub(b2);
		}

		if ((i + 1) % 100 == 0) {
			Matrix::real_t t = net.accuracy(x_batch, t_batch);
			printf("[train %d] %f\n", i + 1, t);
		}
	}

	{
		Matrix x_batch;
		Matrix t_batch;
		for (int j = 0; j < t10k.size(); j++) {
			Matrix x, t;
			t10k.image_to_matrix(j, &x);
			t10k.label_to_matrix(j, &t);
			x_batch.add_rows(x);
			t_batch.add_rows(t);
		}
		Matrix::real_t t = net.accuracy(x_batch, t_batch);
		printf("[t10k] %f\n", t);
	}

	return 0;
}

