#ifndef MNIST_H
#define MNIST_H

#include <vector>
#include <stdlib.h>
#include <stdint.h>
#include <cmath>
#include <map>


struct Matrix {
	unsigned int row = 0;
	unsigned int col = 0;
	std::vector<double> array;

	void make(int r, int c)
	{
		row = r;
		col = c;
		array.clear();
		array.resize(row * col);
	}

	void make(int r, int c, std::initializer_list<double> const &list)
	{
		row = r;
		col = c;
		array.clear();
		array = list;
	}

	void make(int r, int c, double const *data)
	{
		row = r;
		col = c;
		array.clear();
		array.reserve(row * col);
		for (auto i = 0U; i < row; i++) {
			for (auto j = 0U; j < col; j++) {
				array.push_back(*data);
				data++;
			}
		}
	}

	double &at(int r, int c)
	{
		return array[col * r + c];
	}

	double at(int r, int c) const
	{
		return array[col * r + c];
	}

	bool eq(Matrix const &other) const
	{
		auto n = array.size();
		if (n != other.array.size()) return false;
		for (size_t i = 0; i < n; i++) {
			if (array[i] != other.array[i]) return false;
		}
		return true;
	}

	void transpose(Matrix *out) const
	{
		out->make(col, row);
		for (auto r = 0U; r < row; r++) {
			for (auto c = 0U; c < col; c++) {
				out->at(c, r) = at(r, c);
			}
		}
	}

	void transpose()
	{
		Matrix t;
		transpose(&t);
		row = t.row;
		col = t.col;
		array = std::move(t.array);
	}

	void dot(Matrix const &other, Matrix *out) const
	{
		*out = {};
		Matrix const &a = *this;
		Matrix const &b = other;
		auto n = a.col;
		if (n == b.row) {
			auto nrow = a.row;
			auto ncol = b.col;
			out->make(nrow, ncol);
			for (auto col = 0U; col < ncol; col++) {
				for (auto row = 0U; row < nrow; row++) {
					for (auto i = 0U; i < n; i++) {
						out->at(row, col) += a.at(row, i) * b.at(i , col);
					}
				}
			}
		}
	}

	void add(Matrix const &other)
	{
		size_t n = std::min(array.size(), other.array.size());
		for (size_t i = 0; i < n; i++) {
			array[i] += other.array[i];
		}
	}

	void sub(Matrix const &other)
	{
		size_t n = std::min(array.size(), other.array.size());
		for (size_t i = 0; i < n; i++) {
			array[i] -= other.array[i];
		}
	}

	void mul(Matrix const &other)
	{
		size_t n = std::min(array.size(), other.array.size());
		for (size_t i = 0; i < n; i++) {
			array[i] *= other.array[i];
		}
	}

	void mul(double t)
	{
		size_t n = array.size();
		for (size_t i = 0; i < n; i++) {
			array[i] *= t;
		}
	}

	void div(double t)
	{
		size_t n = array.size();
		for (size_t i = 0; i < n; i++) {
			array[i] /= t;
		}
	}

	void sum(Matrix *out) const
	{
		out->make(1, col);
		for (auto r = 0U; r < row; r++) {
			for (auto c = 0U; c < col; c++) {
				out->at(0, c) += at(r, c);
			}
		}
	}

	static double sigmoid(double v)
	{
		return 1 / (1 + exp(-v));
	}

	void sigmoid(Matrix *out) const
	{
		out->make(row, col);
		size_t n = array.size();
		for (size_t i = 0; i < n; i++) {
			out->array[i] = sigmoid(array[i]);
		}
	}

	void sigmoid()
	{
		size_t n = array.size();
		for (size_t i = 0; i < n; i++) {
			array[i] = sigmoid(array[i]);
		}
	}

	void sigmoid_grad(Matrix *out) const
	{
		out->make(row, col);
		size_t n = array.size();
		for (size_t i = 0; i < n; i++) {
			double v = sigmoid(array[i]);
			out->array[i] = (1 - v) * v;
		}
	}

	void softmax(Matrix *out)
	{
		out->make(row, col);
		for (size_t r = 0; r < row; r++) {
			double c = 0;
			for (size_t i = 0; i < col; i++) {
				c = std::max(c, at(r, i));
			}
			std::vector<double> exp_a(col);
			double sum_exp_a = 0;
			for (size_t i = 0; i < col; i++) {
				double v = exp(at(r, i) - c);
				exp_a[i] = v;
				sum_exp_a += v;
			}
			for (size_t i = 0; i < col; i++) {
				out->at(r, i) = exp_a[i] / sum_exp_a;
			}
		}
	}

	double mean_squared_error(const Matrix &other) const
	{
		size_t n = std::min(array.size(), other.array.size());
		double sum = 0;
		for (size_t i = 0; i < n; i++) {
			double d = array[i] - other.array[i];
			sum += d * d;
		}
		return sum / 2;
	}

	double cross_entropy_error(const Matrix &other) const
	{
		size_t n = std::min(array.size(), other.array.size());
		double sum = 0;
		for (size_t i = 0; i < n; i++) {
			sum -= other.array[i] * log(array[i] + 1e-7);
		}
		return sum;
	}

	void add_row(Matrix const &other)
	{
		if (col == 0 && row == 0) {
			make(0, other.col);
		}
		if (col == other.col) {
			row += other.row;
			array.insert(array.end(), other.array.begin(), other.array.end());
		}
	}
};



struct Network {
	std::map<std::string, Matrix> map;

	void parse(char const *begin, char const *end)
	{
		map.clear();
		char const *ptr = begin;
		char const *head = begin;
		std::string name;
		Matrix matrix;
		while (1) {
			int c = 0;
			if (ptr < end) {
				c = (unsigned char)*ptr;
			}
			if (c == ',' || c == '\r' || c == '\n' || c == '[' || c == 0) {
				if (head < ptr) {
					std::string s(head, ptr);
					matrix.array.push_back(strtod(s.c_str(), nullptr));
				}
				ptr++;
				head = ptr;
				if (c == '[' || c == 0) {
					if (!name.empty() && !matrix.array.empty()) {
						auto it = map.insert(map.end(), std::pair<std::string, Matrix>(name, {}));
						std::swap(it->second, matrix);
					}
					if (c == 0) break;
				}
				if (c == '[') {
					int n;
					for (n = 0; ptr + n < end; n++) {
						if (ptr[n] == ']') {
							std::string s(ptr, n);
							std::vector<char> tmp(n);
							n++;
							int row = 0;
							int col = 0;
							int i = sscanf(s.c_str(), "%s %u %u", tmp.data(), &row, &col);
							if (i == 3) {
								matrix = {};
								matrix.row = row;
								matrix.col = col;
								name.assign(tmp.data());
							}
							break;
						}
					}
					ptr += n;
					head = ptr;
				}
			} else {
				ptr++;
			}
		}
	}
};

struct Layer {
	Matrix weight;
	Matrix bias;

	bool load(Network const &network, std::string const &w_name, std::string const &b_name)
	{
		auto i = network.map.find(w_name);
		if (i != network.map.end()) {
			weight = i->second;
		}

		auto j = network.map.find(b_name);
		if (j != network.map.end()) {
			bias = j->second;
		}

		if (weight.row > 0 && weight.col > 0 && weight.row * weight.col == weight.array.size()) {
			// ok
		} else {
			fprintf(stderr, "invalid network weight data: %s\n", w_name.c_str());
			return false;
		}

		if (bias.row > 0 && bias.col > 0 && bias.row * bias.col == bias.array.size()) {
			// ok
		} else {
			fprintf(stderr, "invalid network bias data: %s\n", b_name.c_str());
			return false;
		}

		return true;
	}
};


namespace mnist {

struct DataSet {
	struct Data {
		size_t count = 0;
		int rows = 0;
		int cols = 0;
		std::vector<uint8_t> labels;
		std::vector<std::vector<uint8_t>> images;
	} data;

public:
	bool load(char const *labels_path, char const *images_path);

	bool image_to_matrix(int index, Matrix *out) const;
	void label_to_matrix(int index, Matrix *out) const;
	int label(int index) const;
};


} // namespace mnist

#endif // MNIST_H
